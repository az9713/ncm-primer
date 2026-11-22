"""Project 9: wall-clock of GPT-2 (124M, KV cache, 128 new tokens) vs NCM LM1B small (11 generator passes + decode),
same GPU, same batch, same number of sequences, 128 output tokens each. Speed only: tokenizers and training data differ.
NCM: unigram level-1 draw, generate(T=1.0, top_k=0, top_p=1.0) as config, then decoder argmax (and, separately, the eval
pick_tokens() penalty loop of eval_shard.py:99-124 with the LM1B config values).
GPT-2: model.generate(do_sample=True, max_new_tokens=128, min_new_tokens=128, use_cache=True).
Run 2: the two models are timed in separate phases (NCM loaded, timed, freed; then GPT-2), so each phase has the GPU memory
to itself. The memory cap of ncm_helpers turns an overflow into an OOM, which is recorded per batch size."""
import sys, os, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
REPS = 3
PB = int(os.environ.get("P9_PEN_B", "16"))  # batch for the pick_tokens timing
NSEQ = {int(k): int(v) for k, v in (x.split(":") for x in os.environ.get("P9_NSEQ", "1:64,16:1024,64:1024,256:1024").split(","))}  # batch:sequences
DEC_CHUNK = 16                                    # decoder logits (B,128,30528) fp32: decode in chunks of 16 rows
res = {"env": env_info(), "reps": REPS, "nseq": NSEQ, "ncm": [], "ar": [], "pick_tokens_B16": []}


def timed(fn, nb):
    """Sum of per-batch times; gc.collect() runs between batches, outside the timed region (generate() leaves GPU
    tensors in reference cycles that only the cyclic collector frees; leak_test2.log)."""
    tot = 0.0
    with torch.no_grad():
        for _ in range(nb):
            torch.cuda.synchronize(); t = time.perf_counter()
            fn()
            torch.cuda.synchronize(); tot += time.perf_counter() - t
            gc.collect()
    return tot


# ---------------- phase 1: NCM ----------------
cfg, tk, net, sd = load(cfgpath("lm1b-small"), os.path.join(ART, "lm1b", "small"))
EOS = tk.pre_tokenizer.eos_token_id
p = sd["inference"]["level0_unigram"].float().cpu()
G = cfg.generation
ALLOW = sd["inference"]["token_unigram"].cuda().double()
freq, mu, ADJ = float(G.frequency_penalty), float(G.frequency_deficit), [float(v) for v in str(G.adjacency_penalty).split(",")]
print(f"[mem] after NCM load: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB", flush=True)
res["params"] = {"ncm_generator_incl_shared": sum(q.numel() for q in net.parameters()), "ncm_tokenizer": sum(q.numel() for q in tk.vqvae.parameters())}


def pick_tokens(logits):  # copy of eval_shard.py:99-124 (LM1B config values)
    B, T, V = logits.shape
    lg = logits.double()
    allow = (ALLOW[:V] * T)[None, :]
    al, aw, ad = ADJ[0], int(ADJ[1]), ADJ[2]
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        excess = (cnt - allow).clamp_min(0.0)
        step -= freq * excess
        step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        pp = step.argmax(-1); out[:, t] = pp; hist.append(pp)
        cnt.scatter_add_(1, pp[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    return out


MEMLOG = set()


def _mem(tag, B):
    if B not in MEMLOG:
        print(f"  [mem B={B}] {tag}: allocated {torch.cuda.memory_allocated()/2**20:.0f} MiB, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB, reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB", flush=True)


def ncm_batch(B, penalties=False):
    _mem("before generate", B)
    l1 = torch.multinomial(p, B, replacement=True).view(B, 1).cuda()
    g = net.generate(l1, torch.full((B, 1), EOS, device="cuda"), temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))
    _mem("after generate", B)
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    _mem("after idx_to_zhats", B)
    outs = []
    for s in range(0, B, DEC_CHUNK):
        lg = tk.vqvae.decoder(zl[-1][s:s + DEC_CHUNK])
        outs.append(pick_tokens(lg) if penalties else lg.argmax(-1))
    _mem("after decode", B); MEMLOG.add(B)
    return torch.cat(outs)


for B, N in NSEQ.items():
    nb = N // B
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ncm_batch(B)
        for rep in range(REPS):
            torch.manual_seed(rep)
            t = timed(lambda: ncm_batch(B), nb)
            r = {"B": B, "nseq": N, "rep": rep, "s": t, "s_per_1024": t * 1024 / N, "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20}
            res["ncm"].append(r)
            print(f"NCM B={B:3d} rep {rep} N={N}: {t:.2f}s  per-1024 {r['s_per_1024']:.2f}s  peak reserved {r['peak_reserved_MiB']:.0f} MiB", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        res["ncm"].append({"B": B, "error": f"CUDA OOM under the {MEM_FRACTION} memory cap", "message": str(e)[:400]})
        print(f"NCM B={B:3d}: CUDA OOM under the memory cap {MEM_FRACTION}: {str(e)[:400]}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    dump(os.path.join(OUT, os.environ.get("P9_OUT", "results.json")), res)
for rep in range(REPS):
    try:
        torch.manual_seed(rep)
        t = timed(lambda: ncm_batch(PB, penalties=True), 1024 // PB)
        res["pick_tokens_B16"].append({"rep": rep, "s": t, "B": PB})
        print(f"NCM B={PB} rep {rep} N=1024 with pick_tokens penalties: {t:.2f}s", flush=True)
    except torch.cuda.OutOfMemoryError:
        res["pick_tokens_B16"].append({"rep": rep, "error": "CUDA OOM"}); print(f"pick_tokens B={PB}: CUDA OOM", flush=True); break
dump(os.path.join(OUT, os.environ.get("P9_OUT", "results.json")), res)
del net, tk, sd, ALLOW; gc.collect(); torch.cuda.empty_cache()

# ---------------- phase 2: GPT-2 ----------------
ar_tok = AutoTokenizer.from_pretrained("gpt2")
ar = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
res["params"]["gpt2"] = sum(q.numel() for q in ar.parameters())
print(f"params: GPT-2 {res['params']['gpt2']/1e6:.2f}M, NCM generator {res['params']['ncm_generator_incl_shared']/1e6:.2f}M (module incl. shared), tokenizer {res['params']['ncm_tokenizer']/1e6:.2f}M", flush=True)


def ar_batch(B):
    start = torch.full((B, 1), ar_tok.eos_token_id, device="cuda")
    o = ar.generate(start, attention_mask=torch.ones_like(start), do_sample=True, max_new_tokens=128, min_new_tokens=128,
                    use_cache=True, pad_token_id=ar_tok.eos_token_id)
    assert o.shape[1] == 129, o.shape
    return o


for B, N in NSEQ.items():
    nb = N // B
    try:
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            ar_batch(B)
        for rep in range(REPS):
            torch.manual_seed(rep)
            t = timed(lambda: ar_batch(B), nb)
            r = {"B": B, "nseq": N, "rep": rep, "s": t, "s_per_1024": t * 1024 / N, "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20}
            res["ar"].append(r)
            print(f"AR  B={B:3d} rep {rep} N={N}: {t:.2f}s  per-1024 {r['s_per_1024']:.2f}s  peak reserved {r['peak_reserved_MiB']:.0f} MiB", flush=True)
    except torch.cuda.OutOfMemoryError:
        res["ar"].append({"B": B, "error": f"CUDA OOM under the {MEM_FRACTION} memory cap"})
        print(f"AR  B={B:3d}: CUDA OOM under the memory cap {MEM_FRACTION}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    dump(os.path.join(OUT, os.environ.get("P9_OUT", "results.json")), res)

summ = {}
for B in NSEQ:
    n = [r["s_per_1024"] for r in res["ncm"] if r.get("B") == B and "s" in r]
    a = [r["s_per_1024"] for r in res["ar"] if r.get("B") == B and "s" in r]
    if n and a:
        summ[B] = {"ncm_mean": float(np.mean(n)), "ncm_std": float(np.std(n, ddof=1)) if len(n) > 1 else 0.0,
                   "ar_mean": float(np.mean(a)), "ar_std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
                   "ratio_ar_over_ncm": float(np.mean(a) / np.mean(n))}
        print(f"SUMMARY B={B:3d}: NCM {summ[B]['ncm_mean']:.2f}+/-{summ[B]['ncm_std']:.2f}s  AR {summ[B]['ar_mean']:.2f}+/-{summ[B]['ar_std']:.2f}s per 1024 seqs  AR/NCM {summ[B]['ratio_ar_over_ncm']:.2f}")
    else:
        print(f"SUMMARY B={B:3d}: incomplete (NCM runs {len(n)}, AR runs {len(a)})")
pen = [r["s"] for r in res["pick_tokens_B16"] if "s" in r]
if pen:
    summ["pick_tokens_B16"] = (float(np.mean(pen)), float(np.std(pen, ddof=1)) if len(pen) > 1 else 0.0)
    print(f"SUMMARY NCM B=16 with pick_tokens: {np.mean(pen):.2f}s per 1024")
res["summary"] = summ
dump(os.path.join(OUT, os.environ.get("P9_OUT", "results.json")), res)

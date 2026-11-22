"""Project 6: wall-clock per generator pass and per full generate(), NCM only, RTX 3050 Laptop 4 GB.
Per-pass time = CUDA events from blocks[0] pre-hook to ln_f hook (each end paired with the latest start).
Quality vs cumulative time: Sudoku hard (exact, correct cells) and WMT14 (corpus BLEU) from partial_decodes of the same run."""
import sys, os, pickle, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np, sacrebleu
from transformers import AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
REPS = 5
UNPADDED = False
res = {"env": env_info(), "reps": REPS, "note": "eager path unless kv=True; sampler = config defaults"}
_rp = os.path.join(OUT, "results.json")
if len(sys.argv) > 1 and os.path.exists(_rp):
    _old = json.load(open(_rp)); _old.update({k: v for k, v in res.items() if k in ("env",)}); res = _old  # merge per-task runs


def inputs(task, cfg, tk, sd, B):
    P = int(cfg.model.prefix_len); EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    if task.startswith("sudoku"):
        items = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "puzzles.pickle"), "rb"))
        return None, torch.tensor([list(it)[:91] for it in items[:B]], device="cuda")
    if task in ("lm1b-small", "tinystories"):
        torch.manual_seed(0)
        p = sd["inference"]["level0_unigram"].float().cpu()
        l1 = torch.multinomial(p, B * tk.vqvae.vq.levels[0], replacement=True).view(B, -1).cuda()
        return l1, torch.full((B, 1), EOS, device="cuda")
    if task == "xsum":
        items = pickle.load(open(os.path.join(DATA, "xsum", "passages.pickle"), "rb"))[:B]
        return None, torch.tensor([(list(it)[:P] + [PAD] * P)[:P] for it in items], device="cuda")
    if task == "wmt14":
        srcs = pickle.load(open(os.path.join(DATA, "wmt14", "sources.pickle"), "rb"))
        st = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
        if UNPADDED:  # only sources with >= P tokens: no pad in the prefix
            ids = [st(s)["input_ids"] for s in srcs]
            long_ = [x for x in ids if len(x) >= P]
            res["wmt14_sources_with_at_least_P_tokens"] = len(long_)
            return None, torch.tensor([x[:P] for x in long_[:B]], device="cuda")
        return None, torch.tensor([(st(s)["input_ids"][:P] + [PAD] * P)[:P] for s in srcs[:B]], device="cuda")


def gen_kw(cfg):
    G = cfg.generation
    return dict(temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))


def per_pass(net, l1, prefix, **kw):
    ev = []
    def mark(kind):
        e = torch.cuda.Event(enable_timing=True); e.record(); ev.append((kind, e))
    h1 = net.transformer.blocks[0].register_forward_pre_hook(lambda m, a: mark("s"))
    h2 = net.transformer.ln_f.register_forward_hook(lambda m, a, o: mark("e"))
    try:
        with torch.no_grad():
            codes = net.generate(l1, prefix, **kw)
        torch.cuda.synchronize()
    finally:
        h1.remove(); h2.remove()
    gc.collect()  # generate() leaves GPU tensors in reference cycles (p09 leak_test2.log)
    ms, last = [], None
    for kind, e in ev:
        if kind == "s":
            last = e
        else:
            ms.append(last.elapsed_time(e))
    return codes, ms


def full_time(net, l1, prefix, n=REPS, **kw):
    with torch.no_grad():
        net.generate(l1, prefix, **kw); gc.collect()
        torch.cuda.synchronize(); ts = []
        for _ in range(n):
            t = time.perf_counter(); net.generate(l1, prefix, **kw); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
            gc.collect()  # outside the timed region
    return ts


TASKS = [("sudoku-hard", "sudoku/hard"), ("wmt14", "wmt14"), ("tinystories", "tinystories"), ("lm1b-small", "lm1b/small"), ("xsum", "xsum")]
for task, ck in TASKS:
    if len(sys.argv) > 1 and task not in sys.argv[1:]:
        continue
    cfg, tk, net, sd = load(cfgpath(task), os.path.join(ART, ck))
    kvok = not bool(cfg.model.get("query_level0", False))
    res[task] = {"levels": list(tk.vqvae.vq.levels), "kv_cache_allowed": kvok, "runs": []}
    for B in ((1, 16, 32) if task == "xsum" else (1, 16, 128)):  # xsum B=128 (prefix 512) exceeds 4 GB and spills
        for kv in ([False, True] if kvok else [False]):
            try:
                l1, prefix = inputs(task, cfg, tk, sd, B)
                kw = dict(gen_kw(cfg), use_kv_cache=kv)
                full_time(net, l1, prefix, n=1, **kw)  # warm-up
                codes, ms = per_pass(net, l1, prefix, **kw)
                ts = full_time(net, l1, prefix, **kw)
                L = tk.vqvae.context_length
                r = {"B": B, "kv": kv, "passes": len(ms), "per_pass_ms": ms, "sum_pass_ms": sum(ms),
                     "generate_s_mean": float(np.mean(ts)), "generate_s_std": float(np.std(ts, ddof=1)),
                     "ms_per_output_token": 1000 * float(np.mean(ts)) / (B * L), "peak_mem_MiB": torch.cuda.max_memory_allocated() / 2**20}
                r["peak_reserved_MiB"] = torch.cuda.max_memory_reserved() / 2**20
                if task == "wmt14" and kv:
                    pad = int((prefix == int(cfg.model.prefix_pad_token_id)).any(1).sum())
                    r["rows_with_padded_prefix"] = pad
                    c0 = net.generate(l1, prefix, **dict(kw, use_kv_cache=False))
                    r["code_agreement_kv_vs_eager_pct"] = 100 * float((c0 == codes).float().mean())
                res[task]["runs"].append(r)
                print(f"{task:12s} B={B:3d} kv={kv!s:5s} passes={len(ms):2d} generate {np.mean(ts):.4f}s +/- {np.std(ts, ddof=1):.4f} "
                      f"ms/token {r['ms_per_output_token']:.4f} per-pass ms {[round(v, 2) for v in ms]}"
                      + (f" kv-vs-eager code agreement {r['code_agreement_kv_vs_eager_pct']:.1f}% (padded rows {r['rows_with_padded_prefix']}/{B})" if 'code_agreement_kv_vs_eager_pct' in r else ""))
            except torch.cuda.OutOfMemoryError as e:
                res[task]["runs"].append({"B": B, "kv": kv, "error": "CUDA OOM"}); print(f"{task} B={B} kv={kv}: CUDA OOM")
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    if task == "wmt14":
        UNPADDED = True
        for B in (1, 16):
          try:
            l1, prefix = inputs(task, cfg, tk, sd, B)
            kw = gen_kw(cfg)
            full_time(net, l1, prefix, n=1, **kw); full_time(net, l1, prefix, n=1, use_kv_cache=True, **kw)
            te = full_time(net, l1, prefix, **kw); tkv = full_time(net, l1, prefix, use_kv_cache=True, **kw)
            with torch.no_grad():
                ce = net.generate(l1, prefix, **kw); ck = net.generate(l1, prefix, use_kv_cache=True, **kw)
            agree = 100 * float((ce == ck).float().mean())
            res[task].setdefault("unpadded_kv_check", []).append({"B": B, "eager_s": float(np.median(te)), "kv_s": float(np.median(tkv)),
                "kv_speedup_eager_over_kv": float(np.median(te) / np.median(tkv)), "code_agreement_pct": agree,
                "pads_in_prefix": int((prefix == int(cfg.model.prefix_pad_token_id)).sum())})
            print(f"wmt14 UNPADDED B={B}: eager {np.median(te):.4f}s kv {np.median(tkv):.4f}s speedup {np.median(te)/np.median(tkv):.2f}x code agreement {agree:.1f}% (sources with >=96 tokens: {res['wmt14_sources_with_at_least_P_tokens']})")
          except torch.cuda.OutOfMemoryError:
            res[task].setdefault("unpadded_kv_check", []).append({"B": B, "error": "CUDA OOM"}); print(f"wmt14 UNPADDED B={B}: CUDA OOM")
          torch.cuda.empty_cache()
        UNPADDED = False
    # quality vs cumulative time (B=128 eager run for sudoku; B=32 eager for wmt14: B=128 does not fit under the cap)
    if task in ("sudoku-hard", "wmt14"):
        B = 128 if task == "sudoku-hard" else 32
        res[task]["quality_batch"] = B
        l1, prefix = inputs(task, cfg, tk, sd, B)
        runs_ms = [per_pass(net, l1, prefix, **gen_kw(cfg))[1] for _ in range(5)]
        codes, _ = per_pass(net, l1, prefix, **gen_kw(cfg))
        ms = list(np.median(np.array(runs_ms), axis=0))  # median of 5 runs per pass
        # decode in chunks of 16 rows: WMT14 logits are (B, 64, 151680) fp32, 5 GB at B=128
        chunks = [partial_decodes(tk.vqvae, codes[i:i + 16]) for i in range(0, B, 16)]
        pds = [torch.cat([c[k] for c in chunks]) for k in range(len(chunks[0]))]
        cum = np.cumsum(ms).tolist()
        if task == "sudoku-hard":
            sols = np.array(pickle.load(open(os.path.join(DATA, "sudoku", "hard", "solutions.pickle"), "rb"))[:B])
            CELL = [i for i in range(89) if i % 10 != 9]
            q = [{"level": k + 1, "cum_ms": cum[k], "exact": int((p.cpu().numpy() == sols).all(1).sum()),
                  "mean_correct_cells": float((p.cpu().numpy()[:, CELL] == sols[:, CELL]).sum(1).mean())} for k, p in enumerate(pds)]
        else:
            EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
            refs = pickle.load(open(os.path.join(DATA, "wmt14", "reference_translations.pickle"), "rb"))[:B]
            def tt(row):
                row = row[:row.index(EOS)] if EOS in row else row
                return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)
            # pass 1 is the prefix/level-1 pass; level k code is ready after pass k
            q = [{"level": k + 1, "cum_ms": cum[k], "corpus_bleu": sacrebleu.corpus_bleu([tt(r) for r in p.tolist()], [refs]).score}
                 for k, p in enumerate(pds)]
        res[task]["quality_vs_time"] = q
        print(f"{task} quality vs cumulative ms (B={B}):", [(d["level"], round(d["cum_ms"], 1), d.get("exact", round(d.get("corpus_bleu", 0), 2))) for d in q])
    dump(os.path.join(OUT, "results.json"), res)
    del net, tk; torch.cuda.empty_cache()

"""Project 5: per-level reconstruction accuracy and codebook utilization of every downloaded tokenizer on held-out data."""
import sys, os, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
ONLY = sys.argv[1:]  # optional subset of task names


def rows_text(ids_list, L, EOS, PAD):
    """Format: tokens, then EOS, then PAD up to L; truncated to L (the eval decoder cuts at EOS, eval_shard.py:130-134)."""
    return [((list(ids) + [EOS])[:L] + [PAD] * L)[:L] for ids in ids_list]


def get_rows(task, tk, L):
    pt = tk.pre_tokenizer
    EOS, PAD = pt.eos_token_id, pt.pad_token_id
    if task.startswith("sudoku"):
        d = task.split("-")[1]
        return pickle.load(open(os.path.join(DATA, "sudoku", d, "solutions.pickle"), "rb")), "test solutions (2000 boards, 89 tokens, no padding)"
    if task == "lm1b":
        b = np.fromfile(os.path.join(DATA, "lm1b", "valid_bert_30522.bin"), dtype=np.uint16).astype(np.int64)
        r = b.reshape(-1, 128)
        return r.tolist(), f"valid_bert_30522.bin, all {len(r)} released rows of 128 tokens as stored (rows start with [CLS]=EOS)"
    if task == "tinystories":
        t = pickle.load(open(os.path.join(DATA, "tinystories", "valid_bpe_8192.pickle"), "rb"))
        return rows_text(t, L, EOS, PAD), f"valid_bpe_8192.pickle, all {len(t)} stories; story+EOS, PAD to 128, truncated at 128"
    from transformers import AutoTokenizer
    if task == "xsum":
        refs = pickle.load(open(os.path.join(DATA, "xsum", "reference_summaries.pickle"), "rb"))
        src = AutoTokenizer.from_pretrained("gpt2")
        return rows_text([src(s)["input_ids"] for s in refs], L, EOS, PAD), f"XSum test reference summaries ({len(refs)}), GPT-2 tokens + EOS, PAD to 48"
    if task == "wmt14":
        refs = pickle.load(open(os.path.join(DATA, "wmt14", "reference_translations.pickle"), "rb"))
        src = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        return rows_text([src(s)["input_ids"] for s in refs], L, EOS, PAD), f"WMT14 newstest English references ({len(refs)}), Qwen3 tokens + EOS, PAD to 64"


TASKS = [("sudoku-easy", "sudoku/easy", 128), ("sudoku-medium", "sudoku/medium", 128), ("sudoku-hard", "sudoku/hard", 128),
         ("lm1b", "lm1b/small", 32), ("tinystories", "tinystories", 64), ("xsum", "xsum", 32), ("wmt14", "wmt14", 16)]
CFGN = {"lm1b": "lm1b-small"}
PAPER = {"sudoku-easy": (100.00, 80.73), "sudoku-medium": (100.00, 83.85), "sudoku-hard": (100.00, 80.73),
         "lm1b": (99.86, 89.31), "tinystories": (99.98, 90.12), "xsum": (99.79, 86.20), "wmt14": (98.79, 61.89)}
res = {"env": env_info(), "paper_table1_page": 8}
rp = os.path.join(OUT, "results.json")
if os.path.exists(rp):
    res.update(json.load(open(rp)))
for task, ck, bs in TASKS:
    if ONLY and task not in ONLY:
        continue
    t0 = time.time()
    cfg, tk, _, _ = load(cfgpath(CFGN.get(task, task)), os.path.join(ART, ck), generator=False)
    vq, L, levels = tk.vqvae.vq, tk.vqvae.context_length, list(tk.vqvae.vq.levels)
    K, V = len(levels), list(cfg.tokenizer.vqvae.vqvae_vocab_size)
    rows, desc = get_rows(task, tk, L)
    PAD = tk.pre_tokenizer.pad_token_id
    nk = np.zeros(K); nk_np = np.zeros(K); tot = 0; tot_np = 0; exact = np.zeros(K); seen = [set() for _ in range(K)]
    pd_match = True
    for s in range(0, len(rows), bs):
        x = torch.tensor(rows[s:s + bs], device="cuda")
        m = (x != PAD) if PAD is not None else torch.ones_like(x, dtype=torch.bool)
        with torch.no_grad():
            idx, zh = vq.z_to_idx(tk.vqvae.encode_z(x), return_zhats=True)
            a = 0
            for k, l in enumerate(levels):
                seen[k].update(idx[:, a:a + l].flatten().tolist()); a += l
            if s == 0:  # check partial_decodes() against z_to_idx partial sums and the eval decode path
                pdx = partial_decodes(tk.vqvae, idx)
                _, zl = vq.idx_to_zhats(idx, return_zlevels=True)
                ev = tk.vqvae.decoder(zl[-1]).argmax(-1)
                pd_match = bool(torch.equal(pdx[-1], ev)) and all(torch.equal(pdx[k], tk.vqvae.decoder(zh[k]).argmax(-1)) for k in range(K))
            for k, z in enumerate(zh):
                y = tk.vqvae.decoder(z).argmax(-1)
                eq = (y == x)
                nk[k] += eq.sum().item(); nk_np[k] += (eq & m).sum().item(); exact[k] += eq.all(1).sum().item()
        tot += x.numel(); tot_np += m.sum().item()
    N = len(rows)
    util = [100 * len(seen[k]) / V[k] for k in range(K)]
    r = {"data": desc, "n_rows": N, "L": L, "levels": levels, "codebook_sizes": V,
         "acc_all_pct": [100 * v / tot for v in nk], "acc_nonpad_pct": [100 * v / tot_np for v in nk_np],
         "seq_exact_pct": [100 * v / N for v in exact], "util_per_level_pct": util, "util_mean_pct": float(np.mean(util)),
         "used_codes_total": int(sum(len(x) for x in seen)), "codes_total": int(sum(V)),
         "partial_decodes_matches_eval_and_z_to_idx": pd_match, "paper_acc": PAPER[task][0], "paper_util": PAPER[task][1],
         "seconds": time.time() - t0}
    res[task] = r
    print(f"== {task}: {desc}; partial_decodes check {pd_match}; {r['seconds']:.1f}s")
    print(f"{'lvl':>3} {'l_k':>4} {'acc_all%':>9} {'acc_nonpad%':>11} {'seq_exact%':>10} {'util%':>7}")
    for k in range(K):
        print(f"{k+1:3d} {levels[k]:4d} {r['acc_all_pct'][k]:9.3f} {r['acc_nonpad_pct'][k]:11.3f} {r['seq_exact_pct'][k]:10.2f} {util[k]:7.2f}")
    print(f"final acc_all {r['acc_all_pct'][-1]:.3f} nonpad {r['acc_nonpad_pct'][-1]:.3f} (paper {PAPER[task][0]}); "
          f"util mean {r['util_mean_pct']:.2f} = {r['used_codes_total']}/{r['codes_total']} (paper {PAPER[task][1]})")
    dump(rp, res)
    del tk; torch.cuda.empty_cache()

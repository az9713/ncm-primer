"""Project 4: coarse-to-fine trajectory. Sudoku hard puzzle 0 (= paper Fig. 20) and WMT14 source 1218 (= paper Table 12)."""
import sys, os, pickle, html, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np, sacrebleu

OUT = os.path.dirname(os.path.abspath(__file__))
FIG20 = [16, 15, 16, 21, 19, 25, 28, 44, 55, 62, 64, 81]
res = {"env": env_info(), "paper_fig20_correct_cells": FIG20, "paper_fig20_page": 40, "paper_table12_page": 36}
CELL = [i for i in range(89) if i % 10 != 9]

# ---------- Sudoku hard, puzzle 0 ----------
cfg, tk, net, sd = load(cfgpath("sudoku-hard"), os.path.join(ART, "sudoku", "hard"))
items = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "puzzles.pickle"), "rb"))
sols = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "solutions.pickle"), "rb"))
levels = list(tk.vqvae.vq.levels)
x = torch.tensor([list(items[0])[:91]], device="cuda")
with torch.no_grad():
    codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
A = tk.decode_multiscales(codes)[0]                    # method A: l_k tokens
Bt = [p[0].tolist() for p in partial_decodes(tk.vqvae, codes)]   # method B: 89 tokens
sol, puz = sols[0], list(items[0])[1:90]
corr = [int(sum(b[i] == sol[i] for i in CELL)) for b in Bt]
res["sudoku"] = {"puzzle_index": 0, "n_clues": sum(1 for i in CELL if puz[i] > 0), "puzzle": tk.pre_tokenizer.decode(puz),
                 "solution": tk.pre_tokenizer.decode(sol), "levels": levels,
                 "method_A": A, "method_B": [tk.pre_tokenizer.decode(b) for b in Bt], "correct_cells_B": corr,
                 "abs_diff_vs_fig20": [abs(a - b) for a, b in zip(corr, FIG20)], "exact_match_fig20": corr == FIG20}
print("Sudoku hard puzzle 0, clues:", res["sudoku"]["n_clues"], " puzzle:", res["sudoku"]["puzzle"])
print("correct cells per level (method B):", corr)
print("paper Fig.20                       :", FIG20)
for k in range(len(levels)):
    print(f"  L{k+1:2d} l_k={levels[k]:2d}  A={A[k][:40]:40s}  B={res['sudoku']['method_B'][k]}")


def board_html(toks, sol, puz):
    cells = [toks[i] for i in CELL]; s = [sol[i] for i in CELL]; p = [puz[i] for i in CELL]
    out = "<table class=b>"
    for r in range(9):
        out += "<tr>" + "".join(
            f"<td class='{'ok' if cells[r*9+c]==s[r*9+c] else 'bad'}{' clue' if p[r*9+c]>0 else ''}'>"
            f"{cells[r*9+c] if 1<=cells[r*9+c]<=9 else '?'}</td>" for c in range(9)) + "</tr>"
    return out + "</table>"


del net, tk; torch.cuda.empty_cache()

# ---------- WMT14 source 1218 + per-level corpus BLEU on first N sources ----------
from transformers import AutoTokenizer
cfg, tk, net, sd = load(cfgpath("wmt14"), os.path.join(ART, "wmt14"))
P = int(cfg.model.prefix_len); EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
src_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
srcs = pickle.load(open(os.path.join(DATA, "wmt14", "sources.pickle"), "rb"))
refs = pickle.load(open(os.path.join(DATA, "wmt14", "reference_translations.pickle"), "rb"))
G = cfg.generation
wl = list(tk.vqvae.vq.levels)


def to_text(row):  # eval_shard.py:130-134
    row = list(row); row = row[:row.index(EOS)] if EOS in row else row
    return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)


def gen(batch):
    xx = torch.tensor([(src_tok(s)["input_ids"][:P] + [PAD] * P)[:P] for s in batch], device="cuda")
    with torch.no_grad():
        return net.generate(None, xx, temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))


c = gen([srcs[1218]])
wA = [t for t in tk.decode_multiscales(c, skip_special_tokens=True)[0]]
wB = [to_text(p[0].tolist()) for p in partial_decodes(tk.vqvae, c)]
wbleu = [sacrebleu.sentence_bleu(t, [refs[1218]]).score for t in wB]
res["wmt14_1218"] = {"source": srcs[1218], "reference": refs[1218], "levels": wl, "method_A": wA, "method_B": wB,
                     "sentence_bleu_B": wbleu,
                     "paper_final": "However, there is concern that small and medium-sized enterprises will remain vulnerable to hacking and surveillance.",
                     "paper_final_bleu": 77.8}
print("\nWMT14 source 1218:", srcs[1218]); print("ref:", refs[1218])
for k in range(len(wl)):
    print(f"  L{k+1:2d} l_k={wl[k]:2d} BLEU={wbleu[k]:5.1f}  B: {wB[k]}")
    print(f"              A: {wA[k]}")


def lev(ta, tb):
    d = list(range(len(tb) + 1))
    for i, ca in enumerate(ta, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(tb, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb)); prev, d[j] = d[j], cur
    return d[-1]


N, BS = 500, 32
per = [[] for _ in wl]; ed = [[] for _ in wl]
t0 = time.time()
for s in range(0, N, BS):
    cc = gen(srcs[s:s + BS])
    pds = partial_decodes(tk.vqvae, cc)
    for k, p in enumerate(pds):
        per[k] += [to_text(r) for r in p.tolist()]
    fin = pds[-1].tolist()
    for k, p in enumerate(pds):
        for a_, b_ in zip(p.tolist(), fin):
            ta = a_[:a_.index(EOS)] if EOS in a_ else a_; tb = b_[:b_.index(EOS)] if EOS in b_ else b_
            ed[k].append(lev(ta, tb) / max(len(tb), 1))
cb = [sacrebleu.corpus_bleu(per[k], [refs[:N]]).score for k in range(len(wl))]
res["wmt14_first500"] = {"n": N, "corpus_bleu_per_level": cb, "norm_token_edit_dist_to_final": [float(np.mean(e)) for e in ed],
                         "seconds": time.time() - t0,
                         "sampler": dict(temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))}
print("\nWMT14 first 500: corpus BLEU per level:", [round(v, 2) for v in cb])
print("normalised token edit distance to final per level:", [round(float(np.mean(e)), 3) for e in ed])
dump(os.path.join(OUT, "results.json"), res)

# ---------- HTML ----------
css = ("body{background:#0f172a;color:#e2e8f0;font-family:system-ui,sans-serif;margin:16px;max-width:1100px}"
       ".card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;margin:12px 0;overflow-x:auto}"
       "h1,h2{color:#fb923c}td,th{border:1px solid #334155;padding:4px 6px;vertical-align:top}table{border-collapse:collapse}"
       ".b td{width:18px;height:18px;text-align:center;font-family:monospace;padding:1px}.ok{background:#134e4a;color:#99f6e4}"
       ".bad{color:#fca5a5}.clue{font-weight:bold;text-decoration:underline}.grid{display:flex;flex-wrap:wrap;gap:12px}"
       ".muted{color:#94a3b8}code{color:#2dd4bf}")
S = res["sudoku"]
boards = "".join(f"<div><div class=muted>Level {k+1} ({levels[k]} codes): {S['correct_cells_B'][k]}/81 correct (paper {FIG20[k]})</div>"
                 f"{board_html(Bt[k], sol, puz)}</div>" for k in range(len(levels)))
rows = "".join(f"<tr><td>{k+1}</td><td>{wl[k]}</td><td>{wbleu[k]:.1f}</td><td>{html.escape(wB[k])}</td>"
               f"<td class=muted>{html.escape(wA[k])}</td></tr>" for k in range(len(wl)))
hdr = "".join(f"<th>{k+1}</th>" for k in range(len(wl)))
cbr = "".join(f"<td>{v:.2f}</td>" for v in cb)
edr = "".join(f"<td>{v:.3f}</td>" for v in res["wmt14_first500"]["norm_token_edit_dist_to_final"])
page = ("<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<meta name=color-scheme content=dark><title>NCM trajectory</title><style>{css}</style></head><body>"
        f"<h1>NCM coarse-to-fine trajectory</h1><p class=muted>Generated by viewer.py on {res['env']['time']}, {res['env']['gpu']}. "
        "Method B = full-length partial-sum decode (partial_decodes). Method A = l_k-token decode (decode_multiscales).</p>"
        f"<div class=card><h2>Sudoku hard, puzzle 0 (paper Fig. 20, p.40)</h2><p>Correct cells per level: {S['correct_cells_B']}<br>"
        f"Paper Fig. 20: {FIG20}</p><div class=grid>{boards}</div></div>"
        f"<div class=card><h2>WMT14 source 1218 (paper Table 12, p.36)</h2><p>{html.escape(srcs[1218])}<br>"
        f"<span class=muted>Reference:</span> {html.escape(refs[1218])}</p>"
        f"<table><tr><th>level</th><th>l_k</th><th>BLEU</th><th>method B (64 tokens)</th><th>method A (l_k tokens)</th></tr>{rows}</table></div>"
        f"<div class=card><h2>WMT14 first {N} sources, per level</h2><table><tr><th>level</th>{hdr}</tr>"
        f"<tr><td>corpus BLEU</td>{cbr}</tr><tr><td>norm. edit dist. to final</td>{edr}</tr></table></div></body></html>")
open(os.path.join(OUT, "trajectory.html"), "w", encoding="utf-8").write(page)
print("wrote trajectory.html")

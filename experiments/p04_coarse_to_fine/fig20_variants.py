"""Project 4 follow-up: does a numeric-precision change reproduce the Fig. 20 per-level correct-cell counts of hard puzzle 0?"""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *

OUT = os.path.dirname(os.path.abspath(__file__))
FIG20 = [16, 15, 16, 21, 19, 25, 28, 44, 55, 62, 64, 81]
CELL = [i for i in range(89) if i % 10 != 9]
cfg, tk, net, sd = load(cfgpath("sudoku-hard"), os.path.join(ART, "sudoku", "hard"))
items = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "puzzles.pickle"), "rb"))
sol = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "solutions.pickle"), "rb"))[0]
x = torch.tensor([list(items[0])[:91]], device="cuda")
out = {}
for name, ctx in (("fp32", None), ("tf32", "tf32"), ("autocast_bf16", torch.bfloat16), ("autocast_fp16", torch.float16)):
    torch.backends.cuda.matmul.allow_tf32 = name == "tf32"; torch.backends.cudnn.allow_tf32 = name == "tf32"
    with torch.no_grad():
        if isinstance(ctx, torch.dtype):
            with torch.autocast("cuda", dtype=ctx):
                codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
                pds = partial_decodes(tk.vqvae, codes)
        else:
            codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
            pds = partial_decodes(tk.vqvae, codes)
    corr = [int(sum(int(p[0, i]) == sol[i] for i in CELL)) for p in pds]
    out[name] = {"correct_cells": corr, "equals_fig20": corr == FIG20, "sum_abs_diff": sum(abs(a - b) for a, b in zip(corr, FIG20)),
                 "level1_board": tk.pre_tokenizer.decode(pds[0][0].tolist())}
    print(f"{name:14s} {corr}  sum|diff| vs Fig.20 = {out[name]['sum_abs_diff']}  level1 {out[name]['level1_board'][:29]}")
print(f"{'paper Fig.20':14s} {FIG20}  (Fig. 20 step-1 row 1 reads 967624965)")
dump(os.path.join(OUT, "fig20_variants.json"), out)

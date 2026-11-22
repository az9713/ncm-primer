"""Project 4/7 follow-up: solved rate of all 2,000 hard puzzles under bf16 autocast (the precision that matched Fig. 20 levels 1-8),
compared with the default-precision run (86.25 %) and Table 3 (86.1 % = 1722/2000)."""
import sys, os, pickle, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *

OUT = os.path.dirname(os.path.abspath(__file__))
cfg, tk, net, sd = load(cfgpath("sudoku-hard"), os.path.join(ART, "sudoku", "hard"))
items = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "puzzles.pickle"), "rb"))
sols = pickle.load(open(os.path.join(DATA, "sudoku", "hard", "solutions.pickle"), "rb"))
res = {"env": env_info()}
for name in ("default", "autocast_bf16"):
    solved, t0 = 0, time.time()
    for s in range(0, len(items), 64):  # batch 64: bf16 autocast at batch 128 hit OOM under the 0.8 memory cap
        x = torch.tensor([list(it)[:91] for it in items[s:s + 64]], device="cuda")
        with torch.no_grad():
            if name == "autocast_bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    g = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
                    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                    ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
            else:
                g = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
                _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
        solved += sum(r == list(sol) for r, sol in zip(ids, sols[s:s + 64]))
    res[name] = {"solved": solved, "n": len(items), "pct": 100 * solved / len(items), "seconds": time.time() - t0}
    print(f"{name:14s} solved {solved}/{len(items)} = {100*solved/len(items):.2f}%  ({time.time()-t0:.1f}s)   paper Table 3: 86.1%")
dump(os.path.join(OUT, "bf16_hard_full.json"), res)

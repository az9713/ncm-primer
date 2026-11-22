"""Project 7: Sudoku rule violations, correct cells, changed clues per level (partial decodes), all 3 difficulties."""
import sys, os, pickle, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
CELL = [i for i in range(89) if i % 10 != 9]          # 81 cell positions; 9,19,..,79 are '|' (id 10)
assert len(CELL) == 81
UNITS = ([[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)] +
         [[(br + r) * 9 + bc + c for r in range(3) for c in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)])


def viol(cells):  # cells: (B,81) int array; repeated digits per unit (non-digits ignored)
    v = np.zeros(cells.shape[0], dtype=np.int64)
    for u in UNITS:
        x = cells[:, u]
        for b in range(x.shape[0]):
            d = x[b][(x[b] >= 1) & (x[b] <= 9)]
            v[b] += len(d) - len(set(d.tolist()))
    return v


def text_viol(text):  # spec version (docs/code/projects/07): None if not 81 cells after removing '|'
    g = text.replace("|", "")
    if len(g) != 81:
        return None
    rows = [g[i*9:(i+1)*9] for i in range(9)]
    cols = ["".join(r[j] for r in rows) for j in range(9)]
    boxes = ["".join(rows[r][c] for r in range(br, br+3) for c in range(bc, bc+3)) for br in (0, 3, 6) for bc in (0, 3, 6)]
    return sum(len([ch for ch in u if ch != "_"]) - len(set(ch for ch in u if ch != "_")) for u in rows + cols + boxes)


res = {"env": env_info(), "decode": "partial_decodes (method B, full-length partial sums), argmax", "sampler": "temperature=0.0 top_k=0 top_p=1.0 (config default)"}
for diff in ("hard", "easy", "medium"):
    t0 = time.time()
    cfg, tk, net, sd = load(cfgpath(f"sudoku-{diff}"), os.path.join(ART, "sudoku", diff))
    items = pickle.load(open(os.path.join(DATA, "sudoku", diff, "puzzles.pickle"), "rb"))
    sols = np.array(pickle.load(open(os.path.join(DATA, "sudoku", diff, "solutions.pickle"), "rb")))
    N, B, K = len(items), 128, len(tk.vqvae.vq.levels)
    puz = np.array([list(it)[1:90] for it in items])          # board part of the 91-token prefix
    allv = np.zeros((K, N), np.int64); corr = np.zeros((K, N), np.int64); exact = np.zeros((K, N), bool)
    nond = np.zeros((K, N), np.int64); chg = np.zeros((K, N), np.int64); tv_ok = np.zeros(K, np.int64); tv_sum = np.zeros(K)
    gen_s = 0.0
    for s in range(0, N, B):
        x = torch.tensor([list(it)[:91] for it in items[s:s + B]], device="cuda")
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            codes = net.generate(None, x, temperature=0.0, top_k=0, top_p=1.0)
        torch.cuda.synchronize(); gen_s += time.perf_counter() - t
        for k, p in enumerate(partial_decodes(tk.vqvae, codes)):
            p = p.cpu().numpy(); sl = slice(s, s + p.shape[0])
            cells, scells, pcells = p[:, CELL], sols[sl][:, CELL], puz[sl][:, CELL]
            allv[k, sl] = viol(cells)
            corr[k, sl] = (cells == scells).sum(1)
            exact[k, sl] = (p == sols[sl]).all(1)
            nond[k, sl] = ((cells < 1) | (cells > 9)).sum(1)
            chg[k, sl] = ((pcells > 0) & (cells != pcells)).sum(1)
            for r in p:
                tv = text_viol(tk.pre_tokenizer.decode(r.tolist()))
                if tv is not None:
                    tv_ok[k] += 1; tv_sum[k] += tv
    levels = list(tk.vqvae.vq.levels)
    unsolved = np.where(~exact[-1])[0]
    stop = {}
    for i in unsolved:  # level (1-based) after which violations never fall below the running minimum again
        v = allv[:, i]; stop[int(i)] = int(np.argmin(v) + 1)
    per = [{"level": k + 1, "l_k": levels[k], "mean_violations": float(allv[k].mean()),
            "mean_violations_text_rule": float(tv_sum[k] / max(tv_ok[k], 1)), "boards_parsed_81_cells": int(tv_ok[k]),
            "zero_violation_boards": int((allv[k] == 0).sum()), "mean_correct_cells": float(corr[k].mean()),
            "exact": int(exact[k].sum()), "mean_nondigit_cells": float(nond[k].mean()),
            "mean_changed_clues": float(chg[k].mean())} for k in range(K)]
    res[diff] = {"n": N, "batch": B, "generate_seconds_total": gen_s, "wall_seconds": time.time() - t0,
                 "per_level": per, "final_exact": int(exact[-1].sum()), "final_exact_pct": 100 * float(exact[-1].mean()),
                 "unsolved_n": int(len(unsolved)),
                 "unsolved_final_mean_violations": float(allv[-1, unsolved].mean()) if len(unsolved) else 0.0,
                 "unsolved_min_violation_level_hist": {str(l): int(sum(1 for v in stop.values() if v == l)) for l in range(1, K + 1)},
                 "puzzle0_correct_cells_per_level": corr[:, 0].tolist()}
    print(f"== {diff}: N={N} final exact {exact[-1].sum()}/{N} = {100*exact[-1].mean():.2f}%  generate {gen_s:.1f}s")
    print(f"{'lvl':>3} {'l_k':>3} {'viol':>7} {'viol_txt':>8} {'0viol':>6} {'corr':>6} {'exact':>6} {'nondig':>6} {'chgclue':>7}")
    for r in per:
        print(f"{r['level']:3d} {r['l_k']:3d} {r['mean_violations']:7.2f} {r['mean_violations_text_rule']:8.2f} {r['zero_violation_boards']:6d} "
              f"{r['mean_correct_cells']:6.2f} {r['exact']:6d} {r['mean_nondigit_cells']:6.2f} {r['mean_changed_clues']:7.2f}")
    print("puzzle0 correct cells per level:", corr[:, 0].tolist())
    print("unsolved: level of min violations histogram:", res[diff]["unsolved_min_violation_level_hist"])
    del net, tk; torch.cuda.empty_cache()
dump(os.path.join(OUT, "trace_results.json"), res)

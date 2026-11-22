"""Project 3 follow-up (CPU only): probe.py reseeds the generator with the same seeds for every level-1 code, so sample i of
code A and sample i of code B share the random stream of levels 2-12. Compare TF-cosine of same-seed pairs across codes with
different-seed pairs across codes and with pairs inside one code. Data: samples.txt (first 8 samples of each code, seed 0 batch)."""
import os, re, json, itertools
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
_all = os.path.join(OUT, "samples_all.json")
if os.path.exists(_all):  # all 96 texts per code (probe rerun); index i = seed i//16, row i%16
    groups = [(int(c), t) for c, t in json.load(open(_all, encoding="utf-8")).items()]
else:
    txt = open(os.path.join(OUT, "samples.txt"), encoding="utf-8").read()
    groups = []
    for blk in txt.split("===== code ")[1:]:
        head, body = blk.split("\n", 1)
        groups.append((int(head.split()[0]), [s.strip() for s in body.strip().split("\n---\n")]))
_wc = {}
words = lambda t: _wc[t] if t in _wc else _wc.setdefault(t, re.findall(r"[a-z']+", t.lower()))
vocab = {w: i for i, w in enumerate(sorted({w for _, g in groups for t in g for w in words(t)}))}


def vec(t):
    v = np.zeros(len(vocab)); [v.__setitem__(vocab[w], v[vocab[w]] + 1) for w in words(t)]
    return v / (np.linalg.norm(v) + 1e-9)


keys = [(gi, si) for gi, (_, g) in enumerate(groups) for si, _t in enumerate(g)]
X = np.stack([vec(groups[gi][1][si]) for gi, si in keys])
S = X @ X.T
gid = np.array([k[0] for k in keys]); sid = np.array([k[1] for k in keys])
iu = np.triu_indices(len(keys), 1)
sg = gid[iu[0]] == gid[iu[1]]; ss = sid[iu[0]] == sid[iu[1]]
vals = S[iu]
within = vals[sg]; same_seed = vals[~sg & ss]; diff_seed = vals[~sg & ~ss]
n_s = min(len(g) for _, g in groups)
first4 = lambda t: " ".join(words(t)[:6])
same_open = np.mean([first4(groups[a][1][s]) == first4(groups[b][1][s]) for a, b in itertools.combinations(range(len(groups)), 2) for s in range(n_s)])
diff_open = np.mean([first4(groups[a][1][s]) == first4(groups[b][1][t]) for a, b in itertools.combinations(range(len(groups)), 2) for s in range(n_s) for t in range(n_s) if s != t])
res = {"source": "samples_all.json (96 per code)" if os.path.exists(_all) else "samples.txt (8 per code)", "codes": [c for c, _ in groups], "samples_per_code": n_s,
       "tf_cosine_same_seed_different_code": float(np.mean(same_seed)), "n_same_seed_pairs": len(same_seed),
       "tf_cosine_different_seed_different_code": float(np.mean(diff_seed)), "n_diff_seed_pairs": len(diff_seed),
       "tf_cosine_same_code_different_seed": float(np.mean(within)), "n_within_pairs": len(within),
       "first6words_equal_same_seed_different_code": float(same_open), "first6words_equal_different_seed_different_code": float(diff_open)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(OUT, "seed_effect.json"), "w"), indent=2)

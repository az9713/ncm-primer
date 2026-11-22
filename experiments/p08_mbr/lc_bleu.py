"""Project 8 follow-up (CPU): lowercased corpus BLEU (sacrebleu lowercase=True, as eval_merge.py:27 prints) for greedy, one
sample and MBR M=4/8/12, recomputed from the saved candidates with the same selection rule as mbr.py."""
import os, sys, json, pickle
import numpy as np, sacrebleu
E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
d = json.load(open(os.path.join(E, "p08_mbr", "cands_n3003_t0.8_p0.9.json"), encoding="utf-8"))
refs = pickle.load(open(os.path.join(E, "data", "wmt14", "reference_translations.pickle"), "rb"))[:3003]
greedy, cands = d["greedy"], d["cands"]


def mbr_pick(c):
    sc = [sum(sacrebleu.sentence_bleu(c[i], [x]).score for j, x in enumerate(c) if j != i) for i in range(len(c))]
    return c[int(np.argmax(sc))]


res = {}
for name, hyp in (("greedy", greedy), ("one_sample", cands[0])):
    res[name] = {"bleu": sacrebleu.corpus_bleu(hyp, [refs]).score, "bleu_lc": sacrebleu.corpus_bleu(hyp, [refs], lowercase=True).score}
for M in (4, 8, 12):
    hyp = [mbr_pick([cands[m][i] for m in range(M)]) for i in range(len(greedy))]
    res[f"mbr_M{M}"] = {"bleu": sacrebleu.corpus_bleu(hyp, [refs]).score, "bleu_lc": sacrebleu.corpus_bleu(hyp, [refs], lowercase=True).score}
for k, v in res.items():
    print(f"{k:11s} BLEU {v['bleu']:.2f}  BLEU(lc) {v['bleu_lc']:.2f}")
json.dump(res, open(os.path.join(E, "p08_mbr", "lc_bleu.json"), "w"), indent=2)

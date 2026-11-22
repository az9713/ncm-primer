"""Project 8: MBR decoding for WMT14 De-En.
Greedy = config sampler (T=1.0, top_k=1, top_p=0.0), eager path (use_kv_cache=False, as eval_shard.py:142).
Candidates: M=12 sampled translations per source, one torch seed per (candidate, batch); sampler given by --temp/--top_p.
MBR utility: sentence BLEU (sacrebleu) of a candidate against each other candidate; pick the highest sum.
M=4 and M=8 use the first 4 / 8 candidates of the same pool. Candidates are saved to cands.json.
Usage: python mbr.py --n 3003 --temp 0.8 --top_p 0.9 [--tag NAME]"""
import sys, os, pickle, time, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np, sacrebleu
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3003); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--M", type=int, default=12); ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--top_p", type=float, default=0.9); ap.add_argument("--tag", default="")
a = ap.parse_args()
OUT = os.path.dirname(os.path.abspath(__file__))
tag = a.tag or f"n{a.n}_t{a.temp}_p{a.top_p}"
cfg, tk, net, sd = load(cfgpath("wmt14"), os.path.join(ART, "wmt14"))
P = int(cfg.model.prefix_len)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
src_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
srcs = pickle.load(open(os.path.join(DATA, "wmt14", "sources.pickle"), "rb"))[:a.n]
refs = pickle.load(open(os.path.join(DATA, "wmt14", "reference_translations.pickle"), "rb"))[:a.n]


def to_text(row):  # eval_shard.py:130-134
    row = row[:row.index(EOS)] if EOS in row else row
    return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)


def translate(batch, seed, **kw):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.tensor([(src_tok(s)["input_ids"][:P] + [PAD] * P)[:P] for s in batch], device="cuda")
    with torch.no_grad():
        g = net.generate(None, x, **kw)
        _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
        return [to_text(r) for r in tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()]


def mbr_pick(cands):
    sc = [sum(sacrebleu.sentence_bleu(cands[i], [c]).score for j, c in enumerate(cands) if j != i) for i in range(len(cands))]
    return cands[int(np.argmax(sc))]


G = cfg.generation
t0 = time.time()
greedy = []
for s in range(0, a.n, a.bs):
    greedy += translate(srcs[s:s + a.bs], None, temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))
t_greedy = time.time() - t0
print(f"greedy done {t_greedy:.1f}s", flush=True)
# check: greedy hypotheses must equal the evaluate.py shard hypotheses (project 2, --cfg 0 --bs 16) string for string
ref_pkl = os.path.join(EXP, "p02_guidance_sweep", "wmt_bs16_cfg0", "shard_0.pkl")
greedy_check = None
if os.path.exists(ref_pkl):
    sh = pickle.load(open(ref_pkl, "rb"))
    m = dict(zip(sh["idxs"], sh["hyps"]))
    same = sum(greedy[i] == m.get(i) for i in range(a.n))
    greedy_check = {"equal": same, "n": a.n}
    print(f"greedy vs evaluate.py shard (wmt_bs16_cfg0): {same}/{a.n} strings equal", flush=True)
t0 = time.time()
cands = [[] for _ in range(a.M)]
for bi, s in enumerate(range(0, a.n, a.bs)):
    for m in range(a.M):
        cands[m] += translate(srcs[s:s + a.bs], 100000 * m + bi, temperature=a.temp, top_k=0, top_p=a.top_p)
    if bi % 5 == 0:
        print(f"  candidates batch {bi} ({s}/{a.n}) {time.time() - t0:.0f}s", flush=True)
t_cand = time.time() - t0
json.dump({"greedy": greedy, "cands": cands}, open(os.path.join(OUT, f"cands_{tag}.json"), "w", encoding="utf-8"))
res = {"env": env_info(), "args": vars(a), "greedy_seconds": t_greedy, "candidate_seconds": t_cand, "rows": [],
       "greedy_check_vs_evaluate_shard": greedy_check}


def score(h):
    return sacrebleu.corpus_bleu(h, [refs]).score, sacrebleu.corpus_chrf(h, [refs]).score


b, c = score(greedy)
res["rows"].append({"M": 1, "kind": "greedy", "passes_per_source": 11, "bleu": b, "chrf": c})
print(f"M= 1 greedy           BLEU {b:.2f} chrF {c:.2f}")
b, c = score(cands[0])
res["rows"].append({"M": 1, "kind": "single sample (candidate 0)", "passes_per_source": 11, "bleu": b, "chrf": c})
print(f"M= 1 one sample       BLEU {b:.2f} chrF {c:.2f}")
for M in (4, 8, 12):
    if M > a.M:
        continue
    t1 = time.time()
    hyp = [mbr_pick([cands[m][i] for m in range(M)]) for i in range(a.n)]
    b, c = score(hyp)
    # oracle: best candidate per source by sentence BLEU vs the reference (upper bound of the pool)
    orc = [max((cands[m][i] for m in range(M)), key=lambda h: sacrebleu.sentence_bleu(h, [refs[i]]).score) for i in range(a.n)]
    ob, _ = score(orc)
    res["rows"].append({"M": M, "kind": "MBR", "passes_per_source": 11 * M, "bleu": b, "chrf": c, "oracle_bleu": ob, "select_seconds": time.time() - t1})
    print(f"M={M:2d} MBR ({11*M:3d} passes) BLEU {b:.2f} chrF {c:.2f}  (oracle-by-reference BLEU {ob:.2f}, select {time.time()-t1:.0f}s)")
dump(os.path.join(OUT, f"results_{tag}.json"), res)

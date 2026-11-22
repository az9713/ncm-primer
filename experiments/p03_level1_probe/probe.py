"""Project 3: what does the single level-1 code control in TinyStories?
Fix the level-1 code (10 most frequent codes in the stored unigram table + 10 random codes drawn from the table),
generate 32 samples x 3 seeds per code with the config sampler (T=1.0, top_k=0, top_p=1.0), decode as eval_shard.py does
(no penalties are configured for TinyStories), and compare within-code vs between-code similarity."""
import sys, os, re, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np
from collections import Counter
from rouge_score import rouge_scorer

OUT = os.path.dirname(os.path.abspath(__file__))
cfg, tk, net, sd = load(cfgpath("tinystories"), os.path.join(ART, "tinystories"))
p = sd["inference"]["level0_unigram"].float()
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
B, SEEDS = 16, (0, 1, 2, 3, 4, 5)  # batch 16 x 6 seeds = 96 samples per code (batch 32 hit OOM under the 0.8 memory cap)
G = cfg.generation


def to_text(row):  # eval_shard.py:130-134
    row = row[:row.index(EOS)] if EOS in row else row
    return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)


top = p.topk(10).indices.tolist()
g = torch.Generator().manual_seed(123)
rnd = []
while len(rnd) < 10:
    c = int(torch.multinomial(p.cpu(), 1, generator=g))
    if c not in top and c not in rnd:
        rnd.append(c)
codes = top + rnd
print("active codes in table:", int((p > 0).sum()), "of", p.numel(), "; top-10 mass", round(float(p[top].sum()), 4))
groups = {}
t0 = time.time()
for code in codes:
    texts = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        l1 = torch.full((B, 1), code, device="cuda")
        prefix = torch.full((B, 1), EOS, device="cuda")
        with torch.no_grad():
            gg = net.generate(l1, prefix, temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))
            _, zl = tk.vqvae.vq.idx_to_zhats(gg, return_zlevels=True)
            ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
        texts += [to_text(r) for r in ids]
    groups[code] = texts
gen_s = time.time() - t0
STOP = set("once one the she he they it there his her a an and but then upon so when in on at one day".split())  # names are not stop words
words = lambda t: re.findall(r"[a-z']+", t.lower())


def stats(texts):
    fw = Counter((words(t)[:1] or [""])[0] for t in texts)
    f2 = Counter(" ".join(words(t)[:4]) for t in texts)
    ln = [len(words(t)) for t in texts]
    caps = Counter(w for t in texts for w in re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+", t) if w.lower() not in STOP)
    content = Counter(w for t in texts for w in set(words(t)) if w not in STOP and len(w) > 3)
    return {"first_word_top3": fw.most_common(3), "first4_top3": f2.most_common(3), "mean_words": float(np.mean(ln)),
            "std_words": float(np.std(ln)), "names_top5": caps.most_common(5), "content_words_top8": content.most_common(8)}


summary = {}
for code in codes:
    s = stats(groups[code]); s["p"] = float(p[code]); s["kind"] = "top10" if code in top else "random"
    s["example"] = groups[code][0][:200]; summary[code] = s
    print(f"code {code:4d} ({s['kind']:6s} p={s['p']:.4f}) words {s['mean_words']:.1f}+/-{s['std_words']:.1f} first {s['first_word_top3']} names {s['names_top5'][:3]}")
    print(f"      content {[w for w, _ in s['content_words_top8']]}  ex: {s['example'][:120]!r}")

# similarity: TF cosine over word counts (all pairs) + permutation test; ROUGE-L on sampled pairs
alltexts = [t for c in codes for t in groups[c]]
lab = np.array([i for i, c in enumerate(codes) for _ in groups[c]])
vocab = {w: i for i, w in enumerate(sorted({w for t in alltexts for w in words(t)}))}
X = np.zeros((len(alltexts), len(vocab)), np.float32)
for i, t in enumerate(alltexts):
    for w in words(t):
        X[i, vocab[w]] += 1
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
S = X @ X.T
iu = np.triu_indices(len(alltexts), 1)
same = lab[iu[0]] == lab[iu[1]]
w_cos, b_cos = float(S[iu][same].mean()), float(S[iu][~same].mean())
rng = np.random.default_rng(0)
perm = []
for _ in range(1000):
    pl = rng.permutation(lab); sm = pl[iu[0]] == pl[iu[1]]
    perm.append(S[iu][sm].mean() - S[iu][~sm].mean())
pval = float((np.sum(np.array(perm) >= (w_cos - b_cos)) + 1) / 1001)
sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
idx_same = np.where(same)[0]; idx_diff = np.where(~same)[0]
rs = rng.choice(idx_same, 3000, replace=False); rd = rng.choice(idx_diff, 3000, replace=False)
rl = lambda k: sc.score(alltexts[iu[0][k]], alltexts[iu[1][k]])["rougeL"].fmeasure
w_rl, b_rl = float(np.mean([rl(k) for k in rs])), float(np.mean([rl(k) for k in rd]))
# per-group within-cosine for top10 vs random
per_group_within = {int(c): float(S[np.ix_(lab == i, lab == i)][np.triu_indices((lab == i).sum(), 1)].mean()) for i, c in enumerate(codes)}
first_word_agree = {int(c): summary[c]["first_word_top3"][0][1] / len(groups[c]) for c in codes}
print(f"\nTF-cosine within {w_cos:.4f} between {b_cos:.4f} diff {w_cos - b_cos:.4f} permutation p={pval:.4f} (1000 perms)")
print(f"ROUGE-L (3000 sampled pairs each) within {w_rl:.4f} between {b_rl:.4f}")
print("mean modal-first-word share:", round(float(np.mean(list(first_word_agree.values()))), 3))
res = {"env": env_info(), "codes_top10": top, "codes_random": rnd, "samples_per_code": B * len(SEEDS), "seeds": list(SEEDS),
       "sampler": dict(temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p)),
       "generate_seconds": gen_s, "active_codes": int((p > 0).sum()), "top10_mass": float(p[top].sum()),
       "tf_cosine_within": w_cos, "tf_cosine_between": b_cos, "perm_p": pval, "rougeL_within": w_rl, "rougeL_between": b_rl,
       "per_group_within_cosine": per_group_within, "modal_first_word_share": first_word_agree,
       "groups": {int(c): summary[c] for c in codes}}
dump(os.path.join(OUT, "probe_results.json"), res)
json.dump({str(c): groups[c] for c in codes}, open(os.path.join(OUT, "samples_all.json"), "w", encoding="utf-8"))  # all texts, index i = seed i//B, row i%B
with open(os.path.join(OUT, "samples.txt"), "w", encoding="utf-8") as f:
    for c in codes:
        f.write(f"===== code {c} p={float(p[c]):.5f}\n" + "\n---\n".join(groups[c][:8]) + "\n\n")

"""Fallback scorer: copies of scripts/eval_merge.py genppl_flm() (:110-151) and genppl() (:79-107) with a dtype/device option.
Use only if eval_merge.py cannot hold gpt2-large on the 4 GB GPU. Usage:
  python score_genppl.py --config CFG.yaml --glob "OUT/shard_*.pkl" --metric genppl_flm --dtype fp16 [--limit N]
"""
import argparse, glob, math, os, pickle, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from ncm_helpers import NCM  # puts ncm-main on sys.path
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--glob", required=True)
ap.add_argument("--metric", default=None); ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
ap.add_argument("--device", default="cuda"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
CFG = OmegaConf.load(a.config)
metric = a.metric or str(CFG.eval.metric)
parts = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(a.glob))]
seeds = [s for p in parts for s in p.get("seeds", [])]
toks = [t for p in parts for t in p.get("toks", [])]
raw_ids = [r for p in parts for r in p.get("raw_ids", [])]
hyps_by_part = [h for p in parts for h in p["hyps"]]
dt = torch.float16 if a.dtype == "fp16" else torch.float32
ref = str(CFG.eval.ref_model)
rtok = AutoTokenizer.from_pretrained(ref)
rlm = AutoModelForCausalLM.from_pretrained(ref, dtype=dt).to(a.device).eval()
print(f"[score_genppl] metric={metric} ref={ref} dtype={a.dtype} param_dtype={next(rlm.parameters()).dtype} device={a.device} samples={len(hyps_by_part)} limit={a.limit}")


def sel_for(seed):
    sel = [k for k, s_ in enumerate(seeds) if s_ == seed]
    return sel[:a.limit] if a.limit else sel


per = []
if metric == "genppl_flm":
    gen_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token

    def corpus_ppl(texts):
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        ids, attn = enc["input_ids"].to(a.device), enc["attention_mask"].to(a.device)
        nll_sum = tok_sum = 0.0
        with torch.no_grad():
            for i in range(0, ids.shape[0], 16):
                s_, m_ = ids[i:i + 16], attn[i:i + 16]
                logits = rlm(s_, attention_mask=m_)[0].float().transpose(-1, -2)
                nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
                nll_sum += float((nll * valid).sum()); tok_sum += float(valid.sum())
        return float(np.exp(nll_sum / tok_sum))

    def entropy(ids):
        t = torch.tensor(ids)
        return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())

    for seed in dict.fromkeys(seeds):
        sel = sel_for(seed)
        texts = gen_tok.batch_decode([raw_ids[k] for k in sel])
        per.append((seed, corpus_ppl(texts), float(np.mean([entropy(raw_ids[k]) for k in sel]))))
else:
    from evaluation.gen_ppl import generation_perplexity

    class Cast(torch.nn.Module):  # logits to fp32 before log_softmax, as in the fp32 path
        def __init__(s, m): super().__init__(); s.m = m

        def forward(s, **kw):
            o = s.m(**kw); o.logits = o.logits.float(); return o

    def entropy(ids):
        c, m = Counter(ids), len(ids)
        return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0

    for seed in dict.fromkeys(seeds):
        sel = sel_for(seed)
        gen = [hyps_by_part[k] for k in sel]
        with torch.no_grad():
            ppl = generation_perplexity(Cast(rlm), rtok, [""] * len(gen), gen, batch_size=8, device=a.device).mean_ppl  # eval_merge uses 32; mean_ppl is token-weighted, so batch size does not change it
        per.append((seed, ppl, float(np.mean([entropy(toks[k]) for k in sel if toks[k]]))))
print(f"{'seed':>5} {'genPPL':>8} {'entropy':>8}")
for seed, ppl, ent in per:
    print(f"{seed:5d} {ppl:8.2f} {ent:8.4f}")
g = np.array([p[1] for p in per]); e = np.array([p[2] for p in per])
sd = g.std(ddof=1) if len(g) > 1 else 0.0
print(f"\n{'mean':>5} {g.mean():8.2f} {e.mean():8.4f}   (+/- {sd:.2f} over {len(g)} seeds, {sum(len(sel_for(s)) for s in dict.fromkeys(seeds))} samples)")
if a.device == "cuda":
    print(f"peak GPU memory allocated {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

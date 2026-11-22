# Project 8: add MBR decoding for WMT14

**Goal:** Generate M sampled translations per source and keep the one with the highest mean BLEU against the other candidates. The paper reports BLEU 22.6 without MBR and 25.2 with MBR over 12 candidates (132 NFE).

**Effort:** about 50 lines. **Status:** the selection function was run on three short strings. The WMT14 run was not done.

## Background

- The WMT14 config is greedy (`top_k: 1`, `top_p: 0.0`). Greedy runs give M equal candidates, so you must enable sampling.
- One WMT14 generation is 11 passes. 12 candidates are 12 × 11 = 132 passes, the paper NFE.
- The conditional eval path sets no seed. Set one seed per candidate.
- The sources are strings. The shard tokenizes them with `AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")` and pads to P = 96 (`scripts/eval_shard.py:91`, `:94-96`).

## What exists and what you write

- Exists: `generate()`, the eval decode path, `sacrebleu`.
- You write: the candidate loop, the MBR selection, the corpus score.

## Steps

1. Download WMT14 (1.127 GB) as in [Run an evaluation](../guides/run-an-evaluation.md).
2. Write the script.

   ```python
   import pickle, torch, sacrebleu
   from transformers import AutoTokenizer
   from ncm_helpers import load
   cfg, tk, net, sd = load("config/eval/wmt14.yaml", "artifacts/wmt14")
   P = int(cfg.model.prefix_len)
   EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
   src_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
   srcs = pickle.load(open("data/wmt14/sources.pickle", "rb"))
   refs = pickle.load(open("data/wmt14/reference_translations.pickle", "rb"))

   def to_text(row):
       row = row[:row.index(EOS)] if EOS in row else row
       return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)

   def translate(batch, seed, **kw):
       torch.manual_seed(seed)
       x = torch.tensor([(src_tok(s)["input_ids"][:P] + [PAD] * P)[:P] for s in batch], device="cuda")
       with torch.no_grad():
           g = net.generate(None, x, **kw)
           _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
           return [to_text(r) for r in tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()]

   def mbr_pick(cands):
       score = lambda i: sum(sacrebleu.sentence_bleu(cands[i], [c]).score for j, c in enumerate(cands) if j != i)
       return cands[max(range(len(cands)), key=score)]

   M, BS, N = 12, 32, 500
   hyps = []
   for s in range(0, N, BS):
       batch = srcs[s:s + BS]
       cands = [translate(batch, m, temperature=0.8, top_k=0, top_p=0.9) for m in range(M)]
       hyps += [mbr_pick([cands[m][b] for m in range(M)]) for b in range(len(batch))]
   greedy = sum((translate(srcs[s:s + BS], 0, temperature=1.0, top_k=1, top_p=0.0) for s in range(0, N, BS)), [])
   print("greedy", sacrebleu.corpus_bleu(greedy, [refs[:N]]).score)
   print("MBR   ", sacrebleu.corpus_bleu(hyps, [refs[:N]]).score)
   ```

3. Run with N = 500 first. Then run with all 3,003 sources.
4. Repeat with M = 4 and M = 8.

## Cost and hardware

M = 12 costs 12 times the greedy run, plus M × (M − 1) = 132 sentence-BLEU calls per source. The time was not measured.

## How to measure the result

| M | Passes per source | BLEU | chrF |
|---|-------------------|------|------|
| 1 (greedy) | 11 | | |
| 4 | 44 | | |
| 8 | 88 | | |
| 12 | 132 | | |

Compare with the paper values 22.6 and 25.2. Report the temperature and top-p, because the paper sampler settings for MBR are not in the config.

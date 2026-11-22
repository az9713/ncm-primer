# Project 3: probe what level-1 codes encode

**Goal:** Find out what one level-1 code controls in TinyStories samples. TinyStories has ℓ_1 = 1, so one code out of 4,096 starts each story.

**Effort:** about 30 lines. **Status:** not run.

## Background

For TinyStories, the eval draws the level-1 code from a stored unigram table (`scripts/eval_shard.py:71`, `:156`). The generator then makes levels 2 to 12 from that code and one EOS prefix. If you fix the code, you fix the start of the coarse-to-fine path.

## What exists and what you write

- Exists: `generate(idx, prefix)` accepts your own level-1 codes. The table is in `sd["inference"]["level0_unigram"]`. The `--l0` flag changes the table (`unigram`, `active`, `uniform`, `smooth:<eps>`).
- You write: a loop that fixes the code, generates, decodes and groups the samples.

## Steps

1. Download TinyStories (0.578 GB) as in [Run an evaluation](../guides/run-an-evaluation.md).
2. Run the flag-only part. It compares level-1 draws.

   ```bash
   python evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --l0 uniform --out eval/p3-uniform
   python evaluate.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --l0 smooth:0.1 --out eval/p3-smooth
   ```

3. Write the probe script. It uses `load()` from the [shared setup](index.md).

   ```python
   import torch
   from ncm_helpers import load
   cfg, tk, net, sd = load("config/eval/tinystories.yaml", "artifacts/tinystories")
   p = sd["inference"]["level0_unigram"].float()
   EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
   B = 32

   def to_text(row):                                  # same rule as eval_shard.py:130-134
       row = row[:row.index(EOS)] if EOS in row else row
       return tk.pre_tokenizer.decode([t for t in row if t not in (EOS, PAD)], skip_special_tokens=True)

   groups = {}
   for code in p.topk(10).indices.tolist():          # the 10 most frequent level-1 codes
       torch.manual_seed(0)
       l1 = torch.full((B, 1), code, device="cuda")
       prefix = torch.full((B, 1), EOS, device="cuda")
       with torch.no_grad():
           g = net.generate(l1, prefix, temperature=1.0, top_k=0, top_p=1.0)
           _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
           ids = tk.vqvae.decoder(zl[-1]).argmax(-1).tolist()
       groups[code] = [to_text(r) for r in ids]
   for code, texts in groups.items():
       print(code, round(float(p[code]), 4), texts[0][:120])
   ```

4. For each code, compute simple statistics: the most frequent words, the first word, the mean length, the names of characters.
5. Compare the variation inside one group with the variation between groups.

## Cost and hardware

10 codes × 32 samples × 11 passes. The time was not measured. The model has 88.61M parameters.

## How to measure the result

- Within-group similarity against between-group similarity, for example the mean pairwise ROUGE-L or a sentence-embedding cosine.
- If the within-group similarity is higher, the level-1 code controls a part of the story. Name the part (topic, first word, length) from the statistics of step 4.
- Report the genPPL of the `--l0 uniform` and `--l0 smooth:0.1` runs against the default.

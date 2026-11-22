# Project 10: train a tiny tokenizer

**Goal:** Train a multiscale residual VQ-VAE on Sudoku solutions with the existing training-mode forward. Compare its per-level reconstruction with the released Sudoku tokenizer.

**Effort:** about 60 lines. **Status:** the loop below ran 5 steps on random data on CPU with the `sudoku-hard` config (2026-11-21). The saved file loaded again with `strict=True`. A full training run was not done.

## What exists and what you must write

| Piece | Status | Location |
|-------|--------|----------|
| Training-mode forward with EMA codebooks, fine-level dropout, STE at level 1, final STE | Exists | `models/quant.py:448-531`, `models/vqvae.py:91-95` |
| Commitment and codebook loss terms (`vq_loss`) | Exists | `models/quant.py:519-524` |
| Codebook use per level | Exists, returned by `forward` | `models/quant.py:533-541` |
| k-means codebook start, dead-code restart | Exist, off by default | `models/quant.py:345-378`, `:437-446` |
| Token reconstruction loss | You write | — |
| Optimizer, learning rate, loop | You write | — |
| Data loader | You write | — |
| Loss weights of the paper | Not in the repository | — |
| `fine_drop.dual`, `w_full`, `w_trunc` (LM1B) | In the config, not read by any code | — |

To train a generator too, you need more: the generator loss, the code-corruption step, and the input assembly for `query_level0`. `NextConceptTransformer.forward()` (`models/next_concept.py:376-406`) and `VQVAE.get_ncm_inputs()` (`models/vqvae.py:121-124`) exist. This project does not cover the generator.

## Steps

1. Download the Sudoku solutions for all three difficulties (0.002 GB).

   ```bash
   hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/*/solutions.pickle" --local-dir data
   ```

2. Write the training script. Each solution is a list of 89 ints, the input length of the Sudoku tokenizer.

   ```python
   import sys; sys.path.insert(0, "/path/to/ncm")   # the repository root
   import pickle, torch, torch.nn.functional as F
   from omegaconf import OmegaConf
   from models.vqvae import VQVAE

   cfg = OmegaConf.load("config/eval/sudoku-hard.yaml")
   data = torch.tensor(sum((pickle.load(open(f"data/sudoku/{d}/solutions.pickle", "rb"))
                            for d in ("easy", "medium", "hard")), []))          # (6000, 89)
   vq = VQVAE(**cfg.tokenizer.vqvae).cuda().train()
   opt = torch.optim.AdamW([p for p in vq.parameters() if p.requires_grad], lr=3e-4)
   for step in range(20000):
       x = data[torch.randint(0, len(data), (128,))].cuda()
       logits, vq_loss, util, gutil, _ = vq(x)
       rec = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
       loss = rec + vq_loss                    # 1:1 weights; the paper weights are not in the repository
       opt.zero_grad(); loss.backward(); opt.step()
       if step % 500 == 0:
           print(step, round(rec.item(), 4), round(vq_loss.item(), 4), [round(u, 2) for u in util])
   torch.save({"model": vq.state_dict()}, "my_tokenizer.pt")
   ```

   The codebooks have `requires_grad=False` (`models/quant.py:300`), so the filter keeps them out of the optimizer. The EMA updates them in `forward`.

3. Hold out 500 boards before step 2 if you want a fair test. The step-2 sketch uses all 6,000 for training.
4. Measure the per-level accuracy with the method of [Project 5](05-tokenizer-reconstruction.md), for your tokenizer and for the released `artifacts/sudoku/hard/tokenizer.pt`.
5. Change one setting and train again: `fine_drop.prob` 0.0 against 0.5, or the level schedule. A new schedule needs a `vqvae_vocab_size` list of the same length (`models/quant.py:290-291`).

> **Note:** `codebook_hit_*` buffers count use since construction, so the utilization values only increase (`models/quant.py:522`, `:533-536`).

## Cost and hardware

The Sudoku tokenizer has 14.90M parameters. 20,000 steps were not timed. On CPU, the 5 test steps with batch 8 completed.

## How to measure the result

- Token accuracy per level on held-out boards, for both tokenizers.
- Codebook use per level at the end of training.
- The level at which accuracy passes 0.99, with and without fine-level dropout.

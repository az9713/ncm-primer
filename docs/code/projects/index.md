# Projects

This section gives 10 projects that use the released code and checkpoints. They are in order of effort, from flags only to new training code.

> **Note:** The repository evaluates released models only. It has no optimizer loop, no data loader, no generator loss and no code-corruption code. Projects 1 to 9 use the released weights. Project 10 trains a small tokenizer and needs a training loop that you write.

## What exists and what you must write

| Piece | In the repository | Location |
|-------|-------------------|----------|
| Load tokenizer and generator | Yes | `scripts/eval_shard.py:45-54` |
| Generate codes, level by level | Yes | `models/next_concept.py:521` |
| Decode codes, final level | Yes | `scripts/eval_shard.py:143-144` |
| Decode each level, ℓ_k tokens | Yes | `tokenization/vqvae_tokenizer.py:75-86` |
| Decode each level, L tokens | No; 10 lines | `partial_decodes()` in [Use NCM from Python](../guides/use-ncm-from-python.md) |
| Sampler: temperature, top-k, top-p | Yes | `models/next_concept.py:487-499` |
| Guidance | Yes | `models/next_concept.py:477-485` |
| Metrics: exact, ROUGE, BLEU, genPPL | Yes | `scripts/eval_merge.py` |
| Timing | No | Project 6 |
| MBR decoding | No | Project 8 |
| Autoregressive baseline | No | Project 9 |
| Tokenizer training-mode forward (EMA, fine-level dropout, STE) | Yes | `models/quant.py:448-531` |
| Tokenizer reconstruction loss, optimizer loop, data loader | No | Project 10 |
| Generator training-mode forward | Yes | `models/next_concept.py:376-406` |
| Generator loss, code corruption, query input assembly | No | Not covered |

## Projects in order of effort

| # | Project | New code | Download | Status on 2026-11-21 |
|---|---------|----------|----------|----------------------|
| 1 | [Measure the effect of the decode-time penalties](01-decode-time-penalties.md) | None (flags) | 0.988 GB (LM1B small) or 0.674 GB (XSum) | Not run |
| 2 | [Sweep the guidance weight and the sampler](02-guidance-and-sampler-sweep.md) | None (flags), a shell loop | 1.127 GB (WMT14) or 0.674 GB (XSum) | Not run |
| 3 | [Probe what level-1 codes encode](03-level1-probe.md) | About 30 lines | 0.578 GB (TinyStories) | Not run |
| 4 | [Show the coarse-to-fine trajectory](04-coarse-to-fine-viewer.md) | About 20 lines | any model | Code run on Sudoku |
| 5 | [Measure tokenizer reconstruction per level](05-tokenizer-reconstruction.md) | About 20 lines | any tokenizer | Code run with random weights |
| 6 | [Measure wall-clock time against pass count](06-wall-clock-vs-passes.md) | About 30 lines | any model | Run on Sudoku |
| 7 | [Trace Sudoku rule violations per level](07-sudoku-constraint-trace.md) | About 40 lines | 0.163 GB | Run on Sudoku hard |
| 8 | [Add MBR decoding for WMT14](08-mbr-decoding.md) | About 50 lines | 1.127 GB | Not run |
| 9 | [Time an autoregressive model with a KV cache](09-ar-kv-cache-baseline.md) | About 50 lines | 0.988 GB + GPT-2 | Not run |
| 10 | [Train a tiny tokenizer](10-train-a-tiny-tokenizer.md) | About 60 lines | 0.001 GB (Sudoku solutions) | Loop run 5 steps on random data |

## Shared setup code

Projects 3 to 9 use the `load()` function and the `partial_decodes()` function. Both are in [Use NCM from Python](../guides/use-ncm-from-python.md), steps 2 and 6. Copy them into a file `ncm_helpers.py` outside the repository, and import them:

```python
from ncm_helpers import load, partial_decodes
```

The file must also contain the `sys.path.insert(0, NCM)` line of step 1. The sketches use paths such as `config/eval/...` and `artifacts/...`, so run them with the repository root as the working directory.

Set `PYTHONDONTWRITEBYTECODE=1` to keep `__pycache__` folders out of the repository.

## Rules for all projects

- Report the GPU, the batch size, the config, and the number of inputs with each result.
- Run each sampled measurement with at least 3 seeds, and report the mean and the spread.
- Compare against the config defaults first. Change one setting at a time.

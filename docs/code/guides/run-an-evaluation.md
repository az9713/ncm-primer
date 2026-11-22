# Run an evaluation

Use this guide to evaluate any released config and to change generation settings from the command line.

Prerequisites: [Prerequisites](../getting-started/prerequisites.md). Run all commands from the repository root.

## Status of each command

Only the Sudoku-hard command was run (2026-11-21, result 86.25% solved). The other commands were not run. They use the Hugging Face folder layout that was listed on 2026-11-21.

## Steps

1. Find the folders for your config in this table.

   | Config | `--include` for checkpoints | `--checkpoint-dir` | `--data-dir` |
   |--------|-----------------------------|--------------------|--------------|
   | `sudoku-easy` | `sudoku/easy/*` | `artifacts/sudoku/easy` | `data/sudoku/easy` |
   | `sudoku-medium` | `sudoku/medium/*` | `artifacts/sudoku/medium` | `data/sudoku/medium` |
   | `sudoku-hard` | `sudoku/hard/*` | `artifacts/sudoku/hard` | `data/sudoku/hard` |
   | `xsum` | `xsum/*` | `artifacts/xsum` | `data/xsum` |
   | `wmt14` | `wmt14/*` | `artifacts/wmt14` | `data/wmt14` |
   | `lm1b-small` | `lm1b/tokenizer.pt` and `lm1b/small/*` | `artifacts/lm1b/small` | none |
   | `lm1b-medium` | `lm1b/tokenizer.pt` and `lm1b/medium/*` | `artifacts/lm1b/medium` | none |
   | `lm1b-large` | `lm1b/tokenizer.pt` and `lm1b/large/*` | `artifacts/lm1b/large` | none |
   | `tinystories` | `tinystories/*` | `artifacts/tinystories` | none |
   | `owt` | not released on 2026-11-21 | — | none |

   The LM1B configs name the tokenizer `../tokenizer.pt` (`config/eval/lm1b-small.yaml:4`). Thus `artifacts/lm1b/small` finds the shared file `artifacts/lm1b/tokenizer.pt`.

2. Download the checkpoints. This example is for XSum.

   ```bash
   hf download hazyresearch/ncm-checkpoints --include "xsum/*" --local-dir artifacts
   ```

3. Download the data. Only `xsum`, `wmt14` and `sudoku-*` need data.

   ```bash
   hf download hazyresearch/ncm-datasets --repo-type dataset --include "xsum/*" --local-dir data
   ```

4. Run the evaluation. Status: not run.

   ```bash
   python evaluate.py --config config/eval/xsum.yaml --checkpoint-dir artifacts/xsum --data-dir data/xsum
   ```

   For an unconditional config, omit `--data-dir`. Status: not run.

   ```bash
   python evaluate.py --config config/eval/lm1b-small.yaml --checkpoint-dir artifacts/lm1b/small
   ```

5. Read the result. The merge prints a header line and one result line. The columns for each metric are in [The evaluation pipeline](../concepts/evaluation-pipeline.md).

## Change generation settings

`evaluate.py` passes each unknown flag to every shard (`evaluate.py:17`, `:26-27`). Thus you can add `scripts/eval_shard.py` flags to the `evaluate.py` command.

| Goal | Add |
|------|-----|
| Evaluate the first 200 inputs only | `--n 200` |
| Use GPUs 0 and 2 | `--gpus 0,2` |
| Smaller batches | `--bs 16` |
| Guidance on XSum or WMT14 (the README guided WMT14 row) | `--cfg 1.0` |
| Turn off the frequency penalty | `--freq 0` |
| Turn off the adjacency penalty | `--adj ""` |
| Sample codes with temperature 0.8 and top-p 0.9 | `--temp 0.8 --top_k 0 --top_p 0.9` |
| Uniform level-1 codes in place of the unigram table | `--l0 uniform` |
| Write to a different folder | `--out eval/xsum-cfg1` |

Example (not run):

```bash
python evaluate.py --config config/eval/wmt14.yaml --checkpoint-dir artifacts/wmt14 --data-dir data/wmt14 --cfg 1.0 --out eval/wmt14-cfg1
```

> **Note:** For an unconditional config, `--n` sets the samples per seed, not the total (`scripts/eval_shard.py:84`). With 5 seeds and `--n 64`, the total is 320.

## Verification

1. Examine that `OUT/shard_k.pkl` exists for each GPU.
2. Examine the line `merged N shards -> M <task> outputs`. M must equal the input count: 2,000 (Sudoku), 11,334 (XSum), 3,003 (WMT14), or `n_samples × 5` = 1,280 (unconditional).

## Troubleshooting for this task

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError` for `artifacts/sudoku-hard/tokenizer.pt` | The README path does not match the Hugging Face layout. | Use `artifacts/sudoku/hard` (table in step 1). |
| `no GPU visible; pass --gpus` | `torch.cuda.device_count()` is 0 (`evaluate.py:21-22`). | Install a CUDA build of PyTorch. |
| `shard(s) [0] failed; see .../shard_*.log` | A shard raised an error. | Read `OUT/shard_0.log`. |
| `ValueError: cfg_scale needs prefix_pad_id ...` | `--cfg` on a Sudoku config. | Do not use `--cfg` with Sudoku. |

More issues: [Common issues](../troubleshooting/common-issues.md).

# Command-line reference

This page lists every flag of the three scripts. It tracks `evaluate.py`, `scripts/eval_shard.py` and `scripts/eval_merge.py`.

## `evaluate.py`

Defined at `evaluate.py:10-17`. Unknown flags go to every shard unchanged (`parse_known_args`, `:17`, `:26-27`).

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--config` | path | required | Eval config, for example `config/eval/xsum.yaml`. |
| `--checkpoint-dir` | path | required | Folder with the files named under `checkpoints:`. |
| `--data-dir` | path | `""` | Folder with the files named under `data:`. Not needed for unconditional tasks. |
| `--out` | path | `eval/<task>` | Output folder for `shard_k.log` and `shard_k.pkl`. |
| `--gpus` | string | all visible | Comma-separated GPU ids. One shard per id. |
| `--n` | int | 0 (all) | Evaluate only the first n inputs. |

Exit codes: the script exits with the message `no GPU visible; pass --gpus` when it finds no GPU (`:21-22`). It exits with `shard(s) [...] failed; see <out>/shard_*.log` when a shard fails (`:37-39`). The merge runs with `check=True`, so a merge error stops the script (`:41-42`).

## `scripts/eval_shard.py`

Defined at `scripts/eval_shard.py:9-27`. You can call it directly. It inserts `NCM_ROOT`, or the working directory, into `sys.path` (`:4`).

| Flag | Type | Default | Config field it overrides | Meaning |
|------|------|---------|---------------------------|---------|
| `--config` | path | required | — | Eval config. |
| `--checkpoint-dir` | path | `artifacts/<task>` | — | Checkpoint folder (`:31`). |
| `--data-dir` | path | `$NCM_DATA_ROOT/<task>` or `data/<task>` | — | Data folder (`:32`). |
| `--split` | string | `eval.split` or `test` | `eval.split` | Data split (`:41`). |
| `--n` | int | `eval.n_docs` | `eval.n_docs`, `eval.n_samples` | Inputs (conditional) or samples per seed (unconditional) (`:42`, `:84`). |
| `--bs` | int | `eval.batch_size` | `eval.batch_size` | Batch size (`:43`). |
| `--shard` | int | 0 | — | Index of this shard. |
| `--nshards` | int | 1 | — | Total shards. |
| `--temp` | float | config, else 1.0 | `generation.temperature` | Temperature. |
| `--top_k` | int | config, else 1 | `generation.top_k` | Top-k. |
| `--top_p` | float | config, else 0.0 | `generation.top_p` | Top-p. |
| `--cfg` | float | config, else 0.0 | `generation.cfg_scale` | Guidance weight. |
| `--freq` | float | config, else 0.0 | `generation.frequency_penalty` | Frequency penalty. |
| `--mu` | float | config, else 0.0 | `generation.frequency_deficit` | Frequency deficit bonus. |
| `--adj` | string | config, else empty | `generation.adjacency_penalty` | `"a,w,d"`. `--adj ""` turns it off. |
| `--l0` | string | `unigram` | — | Level-1 draw: `unigram`, `active`, `uniform`, `smooth:<eps>` (`:70-78`). |
| `--out` | path | required | — | Output pickle. |

`--l0` modes, used only when the config has `level0_unigram`:

| Value | Distribution |
|-------|--------------|
| `unigram` | The stored table. |
| `active` | Uniform over codes with probability above 0 in the table (`:72-73`). |
| `uniform` | Uniform over all codes (`:74-75`). |
| `smooth:0.1` | 0.9 × table + 0.1 × uniform (`:76-78`). |

Output pickle keys:

| Key | Mode | Content |
|-----|------|---------|
| `idxs` | both | Input indices of this shard (unconditional: 0 … N−1). |
| `hyps` | both | Decoded output strings. |
| `refs` | both | Reference strings (unconditional: empty strings). |
| `seeds` | unconditional | Seed of each sample. |
| `toks` | unconditional | Output re-encoded by the pre-tokenizer. |
| `raw_ids` | unconditional | Token ids from `pick_tokens()` before EOS cut. |

## `scripts/eval_merge.py`

Defined at `scripts/eval_merge.py:6-9`.

| Flag | Type | Default | Meaning |
|------|------|---------|---------|
| `--config` | path | required | Eval config. Its `eval.metric` selects the scorer (`:154`). |
| `--glob` | string | required | Pattern for the shard pickles, for example `eval/xsum/shard_*.pkl`. |

It stops with `AssertionError: no shard files` when the pattern matches no file (`:13`).

## Environment variables

| Variable | Read at | Meaning |
|----------|---------|---------|
| `NCM_ROOT` | `scripts/eval_shard.py:4`, `scripts/eval_merge.py:4` | Folder added to `sys.path`. Default: the working directory. |
| `NCM_DATA_ROOT` | `scripts/eval_shard.py:32` | Root for the default data folder. Default `data`. |
| `CUDA_VISIBLE_DEVICES` | set by `evaluate.py:30` | One GPU per shard. |
| `TOKENIZERS_PARALLELISM` | set to `false` by `evaluate.py:30` | Turns off Hugging Face tokenizer threads. |

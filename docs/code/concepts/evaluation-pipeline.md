# The evaluation pipeline

Three scripts make one evaluation: `evaluate.py` starts the work, `scripts/eval_shard.py` generates, `scripts/eval_merge.py` scores.

This page tracks `evaluate.py`, `scripts/eval_shard.py`, `scripts/eval_merge.py` and `evaluation/gen_ppl.py`.

## Data flow

```text
evaluate.py --config C --checkpoint-dir CK [--data-dir DD] [extra flags]
   |  one process per GPU id, CUDA_VISIBLE_DEVICES=<id>        evaluate.py:29-35
   v
scripts/eval_shard.py --shard k --nshards n --out OUT/shard_k.pkl
   |  load tokenizer + generator                               eval_shard.py:45-54
   |  read inputs k, k+n, ...  (or seeds k, k+n, ...)          eval_shard.py:82-91
   |  net.generate() -> codes -> decoder logits -> pick_tokens  eval_shard.py:140-146
   v
OUT/shard_k.pkl   {idxs, hyps, refs, [seeds, toks, raw_ids]}   eval_shard.py:162, :169
   |
   v
scripts/eval_merge.py --config C --glob "OUT/shard_*.pkl"     evaluate.py:41-42
   -> prints the metric named by eval.metric                   eval_merge.py:154
```

## Step 1: start the shards

`evaluate.py` reads the GPU list from `--gpus`, or uses all visible GPUs (`evaluate.py:20`). It does not read `eval.n_shards` from the config. It passes unknown flags through to every shard (`:17`, `:26-27`). Each shard writes its log to `OUT/shard_k.log`. If a shard exits with a non-zero code, `evaluate.py` stops and names it (`:37-39`).

> **Warning:** The shards and the merge run with the working directory set to the repository root (`cwd=ROOT`, `evaluate.py:35`, `:42`). The log files open relative to your own working directory (`:31`). Run `evaluate.py` from the repository root, or give absolute paths to `--checkpoint-dir`, `--data-dir` and `--out`.

## Step 2: generate

Each shard does these steps:

1. Read the generation settings. A command-line flag wins over the config value. A missing config value takes the default of `pick()` (`scripts/eval_shard.py:33-40`).
2. Load the tokenizer and the generator with `strict=True` (`:45-54`).
3. Load the frequency allowance and the level-1 unigram table from `generator.pt` if the config names them (`:59-78`).
4. Choose the mode. A task is unconditional when the config has no `data` entry for the split (`:79`).
5. Unconditional: for each seed of this shard, call `torch.manual_seed(seed)`, then make `n_samples` samples in batches. The prefix is one EOS token (`:150-159`).
6. Conditional: read the inputs and the references. Cut or pad each input to P tokens with PAD on the right (`:94-96`). Generate in batches (`:164-169`).
7. Decode: `idx_to_zhats()` → `decoder(z_levels[-1])` → `pick_tokens()` (`:143-144`). Cut at the first EOS when `cut_at_eos` is true (`:127-134`).

> **Note:** The conditional path sets no random seed. With sampling enabled, two runs give different outputs.

## Decode-time token penalties

`pick_tokens()` (`scripts/eval_shard.py:99-124`) works on the decoder token logits, after the generator has finished. Without penalties it returns the argmax (`:100-101`). With penalties it walks the L positions left to right:

- Frequency penalty: subtract `freq × max(count − allowance, 0)` for each token (`:113-115`). The allowance is the stored token unigram × L (`:104`).
- Frequency deficit: add `mu × (L − t)/L × max(allowance − count, 0)` (`:116-117`).
- Adjacency penalty `"a,w,d"`: subtract `a × d^j` from the token chosen j+1 positions back, for the last w positions (`:105-107`, `:118-119`).

| Config | `frequency_penalty` | `frequency_deficit` | `adjacency_penalty` |
|--------|--------------------|---------------------|---------------------|
| `lm1b-*` | 11.5 | 2.0 | `10,4,0.5` |
| `xsum` | 2.0 | 1.0 | `5,4,0.5` |
| others | not set (0) | not set | not set |

This loop makes L small steps on logits. It runs no network pass.

## Step 3: merge and score

`eval_merge.py` loads all `shard_*.pkl` files and sorts the rows by input index (`scripts/eval_merge.py:12-20`). Then it calls the function named by `eval.metric` (`:154`).

| `eval.metric` | Configs | Printed columns | Lines |
|---------------|---------|-----------------|-------|
| `exact` | `sudoku-*` | `solved %`, `solved`, `cell %`, `valid %` | `:58-76` |
| `rouge` | `xsum` | `R-1`, `R-2`, `R-L`, `genPPL` (GPT-2), `words` | `:35-55` |
| `bleu` | `wmt14` | `BLEU`, `BLEU(lc)`, `chrF`, `len ratio`, sacreBLEU signature | `:24-32` |
| `genppl` | `tinystories`, `owt` | per seed: `genPPL`, `entropy`; then mean ± sample std | `:79-107` |
| `genppl_flm` | `lm1b-*` | same columns, FLM protocol | `:110-151` |

`genppl` scores each seed with `generation_perplexity()` under `eval.ref_model` (`gpt2-large`). The perplexity is the exponent of total NLL over total tokens (`evaluation/gen_ppl.py:137-140`). The entropy is the unigram entropy of the re-encoded tokens of each sample, then the mean.

`genppl_flm` decodes the raw token ids with special tokens kept, truncates to 128 GPT-2 tokens, and computes corpus perplexity with batch 16. It counts the first EOS and all non-EOS tokens (`scripts/eval_merge.py:111-135`).

> **Note:** For the genPPL metrics the merge reads the samples in shard order, not sorted order (`hyps_by_part`, `:18`). The seed list makes the groups (`:95-96`).

Related: [Run an evaluation](../guides/run-an-evaluation.md), [Command-line reference](../reference/cli.md).

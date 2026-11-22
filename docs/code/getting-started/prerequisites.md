# Prerequisites

Install these items before you start the [Quickstart](quickstart.md).

## Python 3.10 or newer

The README recommends Python 3.10 or newer.

```bash
python --version
```

The check on 2026-11-21 used Python 3.13.5 on Windows 11.

## PyTorch with CUDA

The eval scripts move every model to the GPU with `.cuda()` (`scripts/eval_shard.py:50-51`). `evaluate.py` stops when it finds no GPU (`evaluate.py:20-22`). A CPU-only installation cannot run the eval scripts.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The output must end with `True`. Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version. The check on 2026-11-21 used `torch 2.11.0+cu128`.

## Python packages

`requirements.txt` lists 9 packages: `torch`, `numpy`, `omegaconf`, `hydra-core`, `transformers>=4.51`, `huggingface_hub[cli]`, `regex`, `rouge_score`, `sacrebleu`.

```bash
pip install -r requirements.txt
```

Verify:

```bash
python -c "import omegaconf, hydra, transformers, sacrebleu, rouge_score; print('ok')"
hf --version
```

## GPU memory

The repository does not state a memory requirement. Measured values do not exist except for one run:

| Config | GPU | Batch size | Result |
|--------|-----|------------|--------|
| `sudoku-hard` | NVIDIA GeForce RTX 3050 Laptop GPU, 4,096 MiB | 128 (config default) | Completed, 2026-11-21 |

The other configs were not run. Their generator sizes (total parameters, counted from the configs with random weights) are 97.07M (`lm1b-small`), 88.61M (`tinystories`), 88.02M (`xsum`), 119.96M (`wmt14`). The `lm1b-medium` and `lm1b-large` generators have 316.11M and 724.29M non-embedding parameters. If a run fails with an out-of-memory error, lower the batch size with `--bs`.

## Disk space

Sizes are from the Hugging Face file lists on 2026-11-21.

| Download | Size |
|----------|------|
| All checkpoints (`hazyresearch/ncm-checkpoints`) | 8.003 GB |
| All datasets (`hazyresearch/ncm-datasets`) | 22.741 GB |
| `sudoku/hard` checkpoints + data | 0.162 GB + 0.0007 GB |
| `xsum` checkpoints + data | 0.654 GB + 0.020 GB |
| `wmt14` checkpoints + data | 1.126 GB + 0.0008 GB |
| `lm1b` shared tokenizer + `small` generator | 0.598 GB + 0.390 GB |
| `lm1b/medium`, `lm1b/large` generators | 1.268 GB, 2.901 GB |
| `tinystories` checkpoints | 0.578 GB |

> **Note:** 18.144 GB of the datasets is the OpenWebText training stream. The unconditional evaluations (LM1B, TinyStories, OWT) need no data files. Download only the folders you need with `--include`.

> **Warning:** On 2026-11-21 the `owt/` checkpoint folder holds only a file `TODO.md`. The OWT generator and tokenizer are not released, so the `owt` config cannot run.

## Network access to Hugging Face

The scripts download pre-tokenizers and scoring models from Hugging Face at run time: `gpt2` (XSum), `Qwen/Qwen3-0.6B` (WMT14 tokenizer), `bert-base-uncased` (LM1B), `gpt2-large` (LM1B, TinyStories, OWT scoring). Sudoku needs none of them.

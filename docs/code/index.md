# NCM evaluation code: documentation

This repository is the released evaluation code for Next-Concept Modeling (NCM), arXiv:2610.14162 (Kumbong, Khare et al., 9 Oct 2026, MIT license). It loads the released checkpoints, generates text or Sudoku solutions level by level, and scores the output.

> **Note:** In the [ncm-primer repository](https://github.com/az9713/ncm-primer), the code is in the folder `ncm-main/`, copied unchanged from [HazyResearch/ncm](https://github.com/HazyResearch/ncm). In these docs, "repository root" means that folder.

> **Note:** The repository contains no training loop, no data pipeline, no generator loss and no code-corruption code. It evaluates released models. To train a model, you must write more code. [Projects](projects/index.md) tells you which parts exist.

## Requirements and external dependencies

Every task, quickstart and project needs the items in this table. Details and verify commands are in [Prerequisites](getting-started/prerequisites.md).

| Requirement | What you need | Notes |
|-------------|---------------|-------|
| Python | 3.10 or newer | Checked with 3.13.5 |
| PyTorch | A CUDA build that matches your driver | Install it before the other packages ([pytorch.org](https://pytorch.org/get-started/locally/)) |
| Python packages | The 9 packages of `requirements.txt`: `torch`, `numpy`, `omegaconf`, `hydra-core`, `transformers>=4.51`, `huggingface_hub[cli]`, `regex`, `rouge_score`, `sacrebleu` | `pip install -r requirements.txt`. No project needs a package outside this list |
| GPU | One NVIDIA GPU with CUDA | Only `sudoku-hard` was run: batch 128 on a 4,096 MiB GPU. Memory for the text models was not measured. Project 10 runs on CPU for a test, but its full run was not timed |
| Checkpoints | Hugging Face repo `hazyresearch/ncm-checkpoints` (8.003 GB for all) | Download only the task you need. The OWT checkpoints are not released: the `owt/` folder holds only `TODO.md` |
| Datasets | Hugging Face dataset `hazyresearch/ncm-datasets` (22.741 GB for all) | The evaluation inputs are about 0.02 GB. Do not download the full repo; see [Common issues](troubleshooting/common-issues.md) |
| Network at run time | Access to the Hugging Face Hub | The scripts download helper models: `gpt2` (XSum), `Qwen/Qwen3-0.6B` (WMT14 tokenizer), `bert-base-uncased` (LM1B), `gpt2-large` (genPPL scoring for LM1B, TinyStories, OWT). Sudoku needs none |
| Code | The folder `ncm-main/` of this repository | Run commands from that folder |

### Requirements for each project

All projects need the base requirements above. The table gives what each project adds. "Helpers" means `load()` and `partial_decodes()` from [Use NCM from Python](guides/use-ncm-from-python.md).

| # | Project | Model download | Helper models downloaded at run time | Hardware | Code you write |
|---|---------|----------------|--------------------------------------|----------|----------------|
| 1 | [Decode-time penalties](projects/01-decode-time-penalties.md) | LM1B small 0.988 GB, or XSum 0.674 GB | LM1B: `bert-base-uncased`, `gpt2-large`. XSum: `gpt2` | GPU | None (flags) |
| 2 | [Guidance and sampler sweep](projects/02-guidance-and-sampler-sweep.md) | WMT14 1.127 GB, or XSum 0.674 GB | WMT14: `Qwen/Qwen3-0.6B`. XSum: `gpt2` | GPU; guidance doubles the batch of each pass | A shell loop |
| 3 | [Level-1 probe](projects/03-level1-probe.md) | TinyStories 0.578 GB | None for the probe; `gpt2-large` only if you score genPPL | GPU | About 30 lines + helpers |
| 4 | [Coarse-to-fine viewer](projects/04-coarse-to-fine-viewer.md) | Any one task; Sudoku hard is the smallest (0.163 GB) | Those of the task you select | GPU | About 20 lines + helpers |
| 5 | [Tokenizer reconstruction per level](projects/05-tokenizer-reconstruction.md) | Any one tokenizer | Those of the task you select | GPU | About 20 lines + helpers |
| 6 | [Wall-clock against pass count](projects/06-wall-clock-vs-passes.md) | Any one task | Those of the task you select | GPU; report its name with the result | About 30 lines + helpers |
| 7 | [Sudoku rule violations per level](projects/07-sudoku-constraint-trace.md) | Sudoku hard 0.163 GB | None | GPU (run on 4,096 MiB) | About 40 lines + helpers |
| 8 | [MBR decoding for WMT14](projects/08-mbr-decoding.md) | WMT14 1.127 GB | `Qwen/Qwen3-0.6B` | GPU; M candidates multiply the generation time by M | About 50 lines + helpers |
| 9 | [Autoregressive KV-cache baseline](projects/09-ar-kv-cache-baseline.md) | LM1B small 0.988 GB | `bert-base-uncased`, `gpt2` (124M) | GPU that holds both models | About 50 lines + helpers |
| 10 | [Train a tiny tokenizer](projects/10-train-a-tiny-tokenizer.md) | Sudoku solutions 0.001 GB (no checkpoint) | None | CPU for a 5-step test; GPU for the full run | About 60 lines: optimizer loop, data loader, reconstruction loss |

---

## Documentation

| Section | What is inside |
|---------|----------------|
| [What is NCM](overview/what-is-this.md) | The two stages, the data flow, the scope of this repository |
| [Key concepts](overview/key-concepts.md) | Definitions of every term in these docs |
| [Prerequisites](getting-started/prerequisites.md) | Python, PyTorch, GPU, packages, disk space |
| [Quickstart](getting-started/quickstart.md) | Download one model and run one evaluation |
| [Onboarding](getting-started/onboarding.md) | The model from zero, with one complete example |
| [The tokenizer](concepts/tokenizer.md) | Multiscale residual VQ-VAE: encode, decode, codebooks |
| [The generator](concepts/generator.md) | `NextConceptTransformer`: attention mask, level-1 modes, `generate()` |
| [The evaluation pipeline](concepts/evaluation-pipeline.md) | `evaluate.py` → `eval_shard.py` → `eval_merge.py`, metrics |
| [Run an evaluation](guides/run-an-evaluation.md) | All tasks, flags, output files |
| [Use NCM from Python](guides/use-ncm-from-python.md) | Load a model, generate, decode, in your own script |
| [Config reference](reference/config.md) | Every field in `config/eval/*.yaml` with the line that reads it |
| [Command-line reference](reference/cli.md) | Every flag of the three scripts |
| [Python API reference](reference/python-api.md) | Classes, methods, signatures, return shapes |
| [Paper and code differences](reference/paper-vs-code.md) | Where the configs and the paper do not agree |
| [System design](architecture/system-design.md) | Components, data flow, cost per pass |
| [Design decisions](architecture/decisions.md) | Three decisions with the reason the code gives |
| [Common issues](troubleshooting/common-issues.md) | Errors, known defects, fixes |
| [Projects](projects/index.md) | 10 projects you can do with this code, in order of effort |

New to NCM? Start with [Onboarding](getting-started/onboarding.md).

## Maintenance rules

- Change a doc in the same commit as the code it describes.
- Each section tracks named files. The config reference tracks `config/eval/*.yaml`, `scripts/eval_shard.py` and `models/*.py`. The CLI reference tracks `evaluate.py`, `scripts/eval_shard.py` and `scripts/eval_merge.py`.
- Every `file.py:N` reference was read on 2026-11-21. After a code change, read the cited lines again and correct the numbers.
- The `.html` files are generated from the `.md` files. Do not edit an `.html` file by hand. After each Markdown change, run this command:

```bash
python md_to_html.py docs/code   # converter not included in this repository
```

- The converter exits 1 and prints `BROKEN` for a link or anchor that does not resolve. Correct the Markdown and run it again.
- Paper numbers and Hugging Face links are dated 2026-11-21. Examine them again when the paper or the collection changes.

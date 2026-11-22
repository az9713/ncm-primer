# NCM evaluation code: documentation

This repository is the released evaluation code for Next-Concept Modeling (NCM), arXiv:2610.14162 (Kumbong, Khare et al., 9 Oct 2026, MIT license). It loads the released checkpoints, generates text or Sudoku solutions level by level, and scores the output.

> **Note:** In the [ncm-primer repository](https://github.com/az9713/ncm-primer), the code is in the folder `ncm-main/`, copied unchanged from [HazyResearch/ncm](https://github.com/HazyResearch/ncm). In these docs, "repository root" means that folder.

> **Note:** The repository contains no training loop, no data pipeline, no generator loss and no code-corruption code. It evaluates released models. To train a model, you must write more code. [Projects](projects/index.md) tells you which parts exist.

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

# Paper and code differences

This page lists the places where the released configs and code do not agree with the paper (arXiv:2610.14162, version of 9 Oct 2026). It tracks `config/eval/*.yaml`, `scripts/eval_shard.py` and `models/quant.py`. The paper values come from the brief for these docs (2026-11-21). They were not compared with the PDF again.

The code does not give a reason for any item below. Do not assume which version produced a paper number.

| Item | Paper | Code | Location |
|------|-------|------|----------|
| XSum level count | K = 8, NFE 8 | 9 levels (`4 6 10 14 20 26 32 40 48`); generation makes 9 passes (counted with random weights) | `config/eval/xsum.yaml:93-102` |
| OWT generator | 12 blocks, width 768, 12 heads, codebooks 4,096, 94.65M parameters | 16 blocks, width 1024, 16 heads, codebooks 8,192 | `config/eval/owt.yaml:7-11`, `:16-47` |
| OWT release | Results reported | The `owt/` checkpoint folder on Hugging Face holds only `TODO.md` (2026-11-21) | Hugging Face `hazyresearch/ncm-checkpoints` |
| Token selection after decoding | Algorithm 2 takes the argmax | LM1B and XSum apply frequency and adjacency penalties: LM1B 11.5, deficit 2.0, `10,4,0.5`; XSum 2.0, deficit 1.0, `5,4,0.5` | `scripts/eval_shard.py:99-123`, `config/eval/lm1b-small.yaml:130-132`, `config/eval/xsum.yaml:110-112` |
| Tokenizer loss with nested cutoff | Terms 2–3 | The terms are averaged over all levels (`:519-524`) before the random cutoff (`:525-530`). The cutoff does not change the loss terms. | `models/quant.py:519-531` |
| LM1B pass count | 12 NFE | 11 generator passes; level 1 is drawn from the unigram table without a pass | `scripts/eval_shard.py:156`, `models/next_concept.py:660` |
| Generator loss, code corruption | Described | Not in the repository. The LM1B config has `aux.code_corruption.*` and `fine_drop.dual`, `w_full`, `w_trunc`, but no code reads them. | grep on 2026-11-21 |

## Results that the code reproduced

| Task | Paper | This repository, 2026-11-21 |
|------|-------|-----------------------------|
| Sudoku hard | 86.1% solved | 86.25% (1725/2000), RTX 3050 Laptop GPU, config defaults |

The other tasks were not run.

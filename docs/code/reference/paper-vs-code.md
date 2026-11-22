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

Runs of 2026-11-21/22 on an RTX 3050 Laptop GPU (4 GB), config defaults unless stated. Details: [experiment report](../../experiments/index.html).

| Task | Paper | This repository |
|------|-------|-----------------|
| Sudoku easy / medium / hard | 99.5 / 96.5 / 86.1 % solved (Table 3) | 99.40 / 96.45 / 86.25 % |
| LM1B small | genPPL 93.05, entropy 4.31 (Table 8) | 94.06 ± 1.42, entropy 4.3138 (with the released penalties); 78.53 without |
| TinyStories | genPPL 15.41, entropy 3.98 (Table 8) | 16.29 ± 0.22, entropy 3.9712 |
| WMT14 | BLEU 22.6; with guidance 23.9 (Table 2) | lowercased 22.62 / 23.86; cased 21.72 / 22.97 |
| WMT14 MBR, M = 12 | BLEU 25.2 (Table 2) | lowercased 24.67; cased 23.81 |
| XSum (first 1,000 documents) | ROUGE-1/2/L 35.0 / 12.8 / 28.0 (Table 2) | 34.63 / 12.84 / 27.78 |
| WMT14 trajectory, source 1218 | Table 12 | equal at all 11 levels |
| Tokenizer accuracy | Table 1 | within 0.02 points for 6 of 7 tokenizers; WMT14 98.507 against 98.79 |

The paper does not state the BLEU casing. The lowercased sacreBLEU column of `scripts/eval_merge.py:27` matches Table 2.

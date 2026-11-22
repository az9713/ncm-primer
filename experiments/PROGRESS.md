# Progress — NCM experiments

All 10 projects run. Final report: `REPORT.md` and `REPORT.html` (assembled by `common/assemble_report.py` from `report_parts/`).

| # | Project | Status | Key number | Evidence |
|---|---------|--------|------------|----------|
| 1 | Penalties | Completed | LM1B default 94.06±1.42 H 4.3138 (paper 93.05/4.31); argmax 78.53/4.1176; XSum 1k R-L 27.78 vs 27.71 | p01_penalties/results.json |
| 2 | Guidance sweep | Completed (subsets stated) | WMT14 lowercased BLEU 22.62 (w0) / 23.86 (w1.0) vs paper 22.6/23.9; cased 21.72/22.97 | p02_guidance_sweep/results.json |
| 3 | Level-1 probe | Completed | TS 16.29±0.22 H 3.9712 (paper 15.41/3.98); code effect +0.018 cosine, seed effect +0.101 | p03_level1_probe/results.json |
| 4 | Coarse-to-fine viewer | Completed | Table 12 reproduced exactly; Fig. 20 levels 1-8 under bf16 | p04_coarse_to_fine/ |
| 5 | Tokenizer recon | Completed | non-pad acc LM1B 99.858, TS 99.980, XSum 99.771, WMT 98.507 (paper 99.86/99.98/99.79/98.79) | p05_tokenizer_recon/ |
| 6 | Wall-clock | Completed within 4 GB | LM1B B16 0.891 s/generate; WMT14 KV unpadded 100 % agreement and 2.35x faster at B16; padded 25.5-37.4 % | p06_wallclock/results.json |
| 7 | Sudoku trace | Completed | 99.40/96.45/86.25 % (paper 99.5/96.5/86.1) | p07_sudoku_trace/results.json |
| 8 | MBR | Completed | M=12 lowercased 24.67 / cased 23.81 (paper 25.2) | p08_mbr/results.json |
| 9 | AR baseline | Completed within 4 GB (B1-B32) | GPT-2/NCM 8.15-8.81 (B1), 4.72 (B4), 2.82 (B8), 1.47 (B16), 1.33 (B32); NCM B>=64 does not fit | p09_ar_baseline/results_all.json, results_b32.json |
| 10 | Train tokenizer | Completed (10,000 steps, real boards) | held-out 100 %, util 89.58 % vs released 80.73 %; 2,500-step runs 69.79 % / 72.40 % | p10_train_tokenizer/train_real10k.json, train_real_10000.log |

ncm-main: git status empty; sha256 manifest of 39 files identical before/after (ncm_main_manifest_before.txt, _after.txt).

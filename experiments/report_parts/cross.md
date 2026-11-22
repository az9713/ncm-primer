## 4. Cross-project findings and limits

### Findings that hold across projects

1. **The released checkpoints reproduce the paper where the setting is fully specified.** Sudoku solved rates are within 0.15 points of Table 3 (project 7). The WMT14 Table 12 trajectory is reproduced exactly, 11 of 11 per-level BLEU values (project 4). Tokenizer accuracy at the last level is within 0.02 points of Table 1 for 6 of 7 tokenizers (project 5). LM1B genPPL is within one seed standard deviation of Table 8 with the released penalties (project 1). WMT14 lowercased BLEU is within 0.04 of Table 2 unguided and guided (project 2).
2. **The paper decodes intermediate levels with full-length partial sums.** Table 12 is reproduced only by method B (`partial_decodes()`), not by the ℓ_k-token method A of `decode_multiscales()` (project 4).
3. **Numeric precision changes intermediate Sudoku decodes.** Autocast in bf16 gives the Fig. 20 correct-cell counts at levels 1 to 8 and the final board (9 of 12 levels equal). Default precision gives 5 of 12 levels equal (levels 3, 4, 5, 11 and 12). On the full hard set, bf16 gives 1727/2000 = 86.35 % and default precision 1725/2000 = 86.25 %, against 86.1 % in Table 3. The paper does not state its inference precision (project 4).
4. **Most quality arrives at the last levels.** On Sudoku, no board is exact before level 12, and mean violations fall from 34.29 to 0.47 in the last level (project 7). Tokenizer accuracy passes 99 % only at the last level for Sudoku, LM1B and XSum (project 5). On WMT14, corpus BLEU rises from 17.26 at level 10 to 20.06 at level 11 on 500 sources (project 4).
5. **The LM1B result depends on a decode-time addition.** The frequency penalty raises genPPL from 78.53 to 94.15. It raises entropy from 4.12 to 4.30. The released default gives 94.06 and 4.31, which matches Table 8 (93.05, 4.31). Algorithm 2 (argmax) does not contain this step (project 1).

### Code defects that affected the work

| Defect | Location | Effect here | Workaround |
|---|---|---|---|
| `reconstruct()` unpacks 3 values, the quantizer returns 5 | `models/vqvae.py:116` | Not called | `z_to_idx(..., return_zhats=True)` and `partial_decodes()` |
| `idx_to_zhats()` returns the same tensor K times | `models/quant.py:651-652` | Not called for per-level decodes | `partial_decodes()`; checked equal to the `z_to_idx` partial sums at all levels for all 7 tokenizers (project 5) |
| KV-cache path drops the prefix pad mask | `models/next_concept.py:654`, `:683` | The eval path does not use the KV cache (`scripts/eval_shard.py:142` passes no `use_kv_cache`; default `False` at `models/next_concept.py:521`), so projects 2 and 8 are not affected. With real WMT14 weights, KV and eager codes agree on 100.0 % of positions with an unpadded prefix and on 37.4 % (batch 1) and 25.5 % (batch 16) with padded prefixes (project 6) | Eager path |
| Default `top_k=50` of `generate()` exceeds the 16 Sudoku codes | `models/next_concept.py:492` | Avoided | `top_k=0` |
| `generate()` leaves GPU tensors in reference cycles (location in the code not identified) | `models/next_concept.py:521-703` (the method) | Live memory grows by about 78 MiB per call at batch 16 until Python's cyclic collector runs; in tight timing loops this caused OOM at batch ≥ 4 (projects 6 and 9) | `gc.collect()` between calls (`p09_ar_baseline/leak_test2.log`) |

### Limits of this replication

- **OWT not released.** The `owt/` checkpoint folder holds only `TODO.md`. The OWT columns of Tables 1 and 8 and the per-level curve of Fig. 16 (p.31) cannot be checked.
- **No diffusion or flow baselines.** The repository contains only NCM. The paper's speed claims are relative to MDLM, DUO, FLM and others on an H100; no baseline was run here. Project 9 adds a GPT-2 timing that the paper does not have.
- **No generator training code.** The repository has no generator loss, code corruption or data loader. Only the tokenizer could be trained (project 10).
- **Hardware.** One RTX 3050 Laptop GPU with 4,095 MiB, shared with a display, Chrome and unrelated CPU jobs. Two consequences: (1) `gpt2-large` fp32 scoring and WMT14 at batch 64 spill to shared memory. I used fp16 scoring (validated) and batch 16 (BLEU change 0.01). (2) Wall-clock numbers carry noise from other processes. I report spreads.
- **Released config differs from the paper in pass counts.** XSum runs 9 passes against the paper's NFE 8. LM1B and TinyStories run 11 generator passes plus a unigram draw for level 1 against the paper's NFE 12 (`scripts/eval_shard.py:156`).
- **Sample sizes.** XSum runs use the first 1,000 of 11,334 test documents. Guidance weights above 1.0 and the sampler sweep use the first 1,000 and 500 WMT14 sources. Each reduction is stated in its project section with the reason (run time on this GPU).

### Paper BLEU casing

The repository prints cased and lowercased BLEU (`scripts/eval_merge.py:26-31`). The paper does not state which it reports. Lowercased BLEU matches Table 2 to within 0.04 for greedy (22.62 against 22.6) and guided decoding (23.86 against 23.9). Cased BLEU is 0.88 and 0.93 lower. Reports of WMT14 BLEU from this code should state the casing.

### Repository integrity

`ncm-main/` was not modified. Before the first run I saved a sha256 manifest of its 39 files (`experiments/ncm_main_manifest_before.txt`). After the last run, `git status --short ncm-main` printed no lines and the new manifest (`ncm_main_manifest_after.txt`) is identical to the first. The output of both checks is in `experiments/ncm_main_check.txt`. No `__pycache__` folder exists in `ncm-main/`.

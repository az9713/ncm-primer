# Next-Concept Modeling Primer

An interactive, single-page primer on **Next-Concept Modeling (NCM)** for a reader new to the subfield. It covers the multiscale residual tokenizer, the next-concept transformer, the attention masks, training, the evidence lower bound, a count of positions against diffusion and next-token prediction, the reported results with their caveats, and the differences between the paper and the released code.

**Read the page:** https://az9713.github.io/ncm-primer/

## Sources

- Paper: Kumbong, Khare, et al., *Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation*, arXiv:2610.14162 (9 Oct 2026). Source of all claims and numbers.
- Code: [HazyResearch/ncm](https://github.com/HazyResearch/ncm), MIT license. A copy is in `ncm-main/`; the page's `file:line` references point into that copy. The two images in `docs/` come from `ncm-main/assets/`.
- Video: Turing Post, "Attention Span" episode on NCM. Used only for attributed analogies and critique.

## Layout

- `docs/index.html` — the page (served by GitHub Pages from `/docs`).
- `ncm-main/` — the released code, unchanged, with its `LICENSE`.

No model was run for this page.

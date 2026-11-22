# Project 9: time an autoregressive model with a KV cache

**Goal:** Measure the wall-clock time of an autoregressive model with a KV cache, with the same batch, length and GPU as NCM. The paper has no such comparison.

**Effort:** about 50 lines. **Status:** run on 2026-11-22, batches 1 to 32 (batch 64 and above do not fit in 4 GB). Full results: [experiment report](../../experiments/index.html).


## Result (run 2026-11-21/22)

- **What it measured:** time of GPT-2 (124M, KV cache) against NCM LM1B small for 1,024 samples of 128 tokens. Speed only; the models differ in data and tokenizer.
- **Result:** GPT-2 takes 8.15 to 8.81 times as long as NCM at batch 1, 4.72 at batch 4, 2.82 at batch 8, 1.47 at batch 16 and 1.33 at batch 32.
- **Result:** at batch 16, NCM takes 61.69 s and GPT-2 90.59 s per 1,024 samples. The eval penalty loop adds 4.93 s to NCM.
- **Unexpected:** NCM batch 64 does not fit in 4 GB. Repeated `generate()` calls need `gc.collect()` to free memory.
- **Lesson:** NCM's lead over a KV-cache autoregressive model falls as the batch grows. The paper has no autoregressive timing to compare.

## Background

The paper compares NCM with diffusion models: on LM1B, 21× less wall-clock than DUO (1,024 sequences, batch 16, one H100). An autoregressive model with a KV cache makes 128 small passes for 128 tokens. Each pass processes one new token per row. NCM makes 11 passes over 1 + ℓ_k rows. The comparison shows which cost is lower on your GPU.

The repository has no autoregressive model. The models below are not trained on the same data, and they use different tokenizers. This project compares speed only. A quality comparison needs a model trained on the same data.

## What exists and what you write

- Exists: NCM LM1B small (97.07M generator parameters), the timing method of [Project 6](06-wall-clock-vs-passes.md).
- You write: the autoregressive timing with `transformers`, the NCM timing loop.

## Steps

1. Download LM1B small (0.988 GB). `gpt2` (124M parameters) downloads at run time.
2. Time NCM: 1,024 sequences in batches of 16, with the unigram level-1 draw.

   ```python
   import time, torch
   from ncm_helpers import load
   cfg, tk, net, sd = load("config/eval/lm1b-small.yaml", "artifacts/lm1b/small")
   EOS = tk.pre_tokenizer.eos_token_id
   p = sd["inference"]["level0_unigram"].float().cpu()
   B, N = 16, 1024
   torch.cuda.synchronize(); t = time.perf_counter()
   with torch.no_grad():
       for _ in range(N // B):
           l1 = torch.multinomial(p, B, replacement=True).view(B, 1).cuda()
           g = net.generate(l1, torch.full((B, 1), EOS, device="cuda"), temperature=1.0, top_k=0, top_p=1.0)
           _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
           ids = tk.vqvae.decoder(zl[-1]).argmax(-1)
   torch.cuda.synchronize(); print("NCM seconds", time.perf_counter() - t)
   ```

   This loop uses argmax for the final tokens. The eval uses `pick_tokens()` with penalties, which adds a 128-step loop on logits per batch. Time both if you report both.

3. Time GPT-2 with a KV cache: 128 new tokens, batch 16.

   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   tok = AutoTokenizer.from_pretrained("gpt2")
   ar = AutoModelForCausalLM.from_pretrained("gpt2").cuda().eval()
   start = torch.full((B, 1), tok.eos_token_id, device="cuda")
   torch.cuda.synchronize(); t = time.perf_counter()
   with torch.no_grad():
       for _ in range(N // B):
           ar.generate(start, do_sample=True, max_new_tokens=128, min_new_tokens=128,
                       use_cache=True, pad_token_id=tok.eos_token_id)
   torch.cuda.synchronize(); print("AR seconds", time.perf_counter() - t)
   ```

4. Do a warm-up call for each model before the timed loop.
5. Repeat with batch sizes 1, 16, 64 and 256.

## Cost and hardware

Both models fit on one GPU. The time was not measured.

## How to measure the result

| Batch | NCM s / 1,024 seqs | AR s / 1,024 seqs | Ratio AR / NCM |
|-------|--------------------|-------------------|----------------|
| 1 | | | |
| 16 | | | |
| 64 | | | |
| 256 | | | |

- Report the GPU, the PyTorch version, and the parameter counts: NCM 97.07M generator + 111.40M tokenizer, GPT-2 124M.
- State that the two models have different tokenizers: 128 BERT tokens for NCM, 128 GPT-2 tokens for GPT-2.

# Design decisions

This page records three decisions for which the code gives a reason in a comment. The alternatives are the ones that the comment names. The repository does not record other alternatives.

## ADR 001: straight-through estimator at level 1 only

**Status:** Accepted (as implemented in `models/quant.py:503-505`).

### Context
The quantizer selects codes by argmin, which has no gradient. A straight-through estimator copies the gradient past the quantization step.

### Decision
Apply the straight-through estimator only at level 1, and only when `learned_l1_sampling` is true.

### Alternatives considered
- Straight-through at every level. The comment at `models/quant.py:503` says that at higher levels it "diverges within ~100 steps".

### Rationale
Level 1 has a learned resampler (`learned_down_l1`), which needs a gradient. The other levels use fixed area pooling.

### Trade-offs
The encoder gets gradient through level 1 and through the final STE on the sum (`:531`). It gets no per-level gradient at levels 2 to K.

### Consequences
Do not add STE at higher levels in a training script without a test of stability.

## ADR 002: no learned scale in the pre-quantization norm

**Status:** Accepted (as implemented in `models/vqvae.py:59-67`).

### Context
The encoder output goes through a norm before quantization.

### Decision
Use `GroupNorm(..., affine=False)` or `LayerNorm(..., elementwise_affine=False)`.

### Alternatives considered
- A norm with a learned scale gamma. The comment says that the squared distance ‖z − e‖² is not scale-invariant, so a learned gamma "would inflate without bound".

### Rationale
Without gamma, the encoder output has a fixed scale. The codebook vectors can follow it.

### Trade-offs
The model cannot learn a global scale for the encoder output.

### Consequences
A training script must not add `affine=True` here.

## ADR 003: keep the tokenizer out of the generator state dict

**Status:** Accepted (as implemented in `models/next_concept.py:80-81`).

### Context
The generator needs the tokenizer codebooks and resampling functions at each level.

### Decision
Store the `VQVAE` in a one-element tuple, `self.vqvae_local = (vqvae_local,)`.

### Alternatives considered
- Store it as a normal submodule. The comment says the tuple keeps the frozen VQ-VAE "out of the module tree and the state dict".

### Rationale
The tokenizer is frozen and is saved in its own file, `tokenizer.pt`.

### Trade-offs
`net.cuda()` does not move the tokenizer. `scripts/eval_shard.py:50` moves it separately with `tk.vqvae.cuda()`. `net.parameters()` does not include tokenizer weights.

### Consequences
Move the tokenizer to the device yourself. Load `tokenizer.pt` before you build the generator, because some heads copy the codebooks at construction (`_TiedCodebookHead`, `models/next_concept.py:17-25`).

# Onboarding

This page explains NCM from zero. It needs no earlier knowledge of NCM. It assumes that you know what a transformer and a token are.

## What does an autoregressive model do?

An autoregressive model writes token 1, then token 2, then token 3. Each token needs one network pass. A 128-token sentence needs 128 passes. The passes cannot run at the same time, because each pass needs the earlier tokens.

## What does NCM do differently?

NCM writes the whole sequence many times, each time at a longer length. The first version has 1 position (LM1B). The next versions have 8, 16, 24, … positions. The last version has 128 positions, one for each token. Each version is a "level". One pass makes one complete level. All positions in a level are made in parallel.

So LM1B needs 12 levels and 11 or 12 passes, not 128 passes.

## What is at one position of a coarse level?

At a coarse level, a position does not hold a token. It holds a code: an integer that selects one vector from a codebook. At level 1 of LM1B, one code must describe the whole sentence. At level 2, 8 codes describe 8 parts of the sentence.

The tokenizer makes these codes. It is a VQ-VAE (vector-quantized autoencoder) with K levels.

## How does the tokenizer make codes?

1. The encoder changes each of the L tokens into a vector of size D. In all released configs, the encoder is an embedding lookup only (0 transformer blocks, `models/vqvae.py:41`).
2. A GroupNorm without learned scale normalizes the vectors (`models/vqvae.py:59-67`).
3. For level 1, the quantizer shortens the L vectors to ℓ_1 vectors. It finds the nearest codebook vector for each (`models/quant.py:587-600`).
4. It lengthens the selected vectors back to L, applies Phi, and subtracts them from the input. The remainder is the residual.
5. It repeats steps 3 and 4 on the residual for levels 2 to K. Level K has ℓ_K = L positions, so it needs no shortening.

The sum of all K levels is z_hat. The decoder changes z_hat back into L tokens.

## How does the generator use the codes?

The generator gets the sum of levels 1 to k, shortened to ℓ_{k+1} positions. From this input, it predicts the ℓ_{k+1} codes of level k+1. It sees the prefix too: the source text, the puzzle, or one EOS token.

The generator never sees the actual tokens of the output. It sees only the running sum of codebook vectors.

## How does level 1 start?

Level 1 has no earlier level. The configs use three different methods:

| Method | Configs | What happens |
|--------|---------|--------------|
| Unigram table | `lm1b-*`, `tinystories`, `owt` | The eval script draws level-1 codes from a stored frequency table. No network pass. |
| Learned queries | `xsum`, `sudoku-*` | Learned vectors take the place of the input. One pass makes level 1. |
| Prefix head | `wmt14` | A separate head reads the last prefix positions. One pass makes level 1. |

## A complete example: one Sudoku board

This example follows one puzzle through `python evaluate.py --config config/eval/sudoku-hard.yaml ...`.

The puzzle is 91 tokens: `<BOS>`, 81 cells with `_` for a blank, 8 `|` row separators, and `<BOS>` again. This is the first test puzzle, as decoded on 2026-11-21:

```text
<BOS>_216_7_85|76____2__|__3__27__|_523_____|_________|8_67_____|2___84_36|6_81__4__|_4__9___7<BOS>
```

The shard puts the 91 tokens into the prefix (`prefix_len: 91`). The generator runs 12 passes. Pass 1 uses the learned queries and makes 1 code. Passes 2 to 12 make 2, 4, 6, 9, 13, 18, 26, 37, 50, 67 and 89 codes. The decoder changes the sum of all 322 codes into 89 tokens. For this puzzle the result is the correct solution:

```text
921647385|764835291|583912764|452369178|179458623|836721549|217584936|698173452|345296817
```

You can decode the partial sum after each level. On the first 128 hard puzzles, the mean count of Sudoku rule violations per board was 95.19 after level 1 and 35.20 after level 11. After level 12 it was 0.55. [Project 7](../projects/07-sudoku-constraint-trace.md) shows the method.

## Why does the code work this way?

**Why does each level have its own codebook and head?** Each level holds a different kind of residual. A level-1 code describes a whole sequence, and a level-12 code describes one token. The code gives each level its own codebook (`models/quant.py:294-296`) and its own output head (`models/next_concept.py:233-237`). Thus a released checkpoint works with one level schedule only.

**Why does the mask block other levels?** In training, all levels go through the generator in one pass. Each level must predict its codes from its own input only. The block-diagonal mask (`models/next_concept.py:274`) makes training match generation, where one pass sees one level.

**Why are the codebooks not trained by gradients?** The code sets `requires_grad_(False)` on the codebooks (`models/quant.py:297-300`). It updates them with an EMA of the assigned encoder vectors (`:485-498`). The code gives no reason. EMA updates are a common VQ-VAE method to keep codebook vectors near the encoder outputs.

## Where to go next

1. [What is NCM](../overview/what-is-this.md) for the system diagram.
2. [The tokenizer](../concepts/tokenizer.md) and [the generator](../concepts/generator.md) for the details.
3. [Use NCM from Python](../guides/use-ncm-from-python.md) to call the models in your own script.
4. [Projects](../projects/index.md) for 10 experiments.

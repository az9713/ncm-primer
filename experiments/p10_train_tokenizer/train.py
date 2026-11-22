"""Project 10: train a Sudoku multiscale residual VQ-VAE with the repository's training-mode forward (models/quant.py:448-549).
Written here: data loader, token reconstruction loss, optimizer loop. Settings follow paper Table 5 (p.23) where the budget allows:
AdamW(0.9, 0.99), wd 0.01, peak lr 5e-4, warmup 500, cosine to floor 5e-5, grad clip 1, EMA 0.99 (config), nested dropout 0.5 (config).
Loss = cross-entropy + vq_loss (1:1; the paper loss weights are not in the repository).
Data: Sudoku solutions of all three difficulties (6000 boards). 1000 boards held out (fixed seed).
--aug: each training board passes through a random validity-preserving transform (digit relabel, row/col swaps within bands,
band/stack swaps, transpose). Usage: python train.py --steps 10000 --bs 1024 --tag NAME [--aug] [--fine_drop 0.5]"""
import sys, os, pickle, time, math, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
import numpy as np, torch.nn.functional as F
from models.vqvae import VQVAE

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--bs", type=int, default=1024)
ap.add_argument("--tag", default="run"); ap.add_argument("--aug", action="store_true")
ap.add_argument("--fine_drop", type=float, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-batches per optimizer step")
a = ap.parse_args()
OUT = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(a.seed); np.random.seed(a.seed)

cfg = OmegaConf.load(cfgpath("sudoku-hard"))
if a.fine_drop is not None:
    cfg.tokenizer.vqvae.vector_quantizer_config.aux.fine_drop.prob = a.fine_drop
sols = np.array(sum((pickle.load(open(os.path.join(DATA, "sudoku", d, "solutions.pickle"), "rb")) for d in ("easy", "medium", "hard")), []))
perm = np.random.default_rng(1234).permutation(len(sols))
test, train = sols[perm[:1000]], sols[perm[1000:]]
CELL = np.array([i for i in range(89) if i % 10 != 9])


def augment(b):  # b: (B,89) -> valid transformed grids
    B = b.shape[0]; g = b[:, CELL].reshape(B, 9, 9)
    rng = np.random.default_rng()
    out = np.empty_like(g)
    for i in range(B):
        x = g[i]
        relabel = np.concatenate([[0], rng.permutation(9) + 1]); x = relabel[x]
        rows = np.concatenate([bnd * 3 + rng.permutation(3) for bnd in rng.permutation(3)])
        cols = np.concatenate([bnd * 3 + rng.permutation(3) for bnd in rng.permutation(3)])
        x = x[rows][:, cols]
        if rng.random() < 0.5:
            x = x.T
        out[i] = x
    r = b.copy(); r[:, CELL] = out.reshape(B, 81)
    return r


def valid(b):
    g = b[:, CELL].reshape(-1, 9, 9)
    ok = np.ones(len(g), bool)
    for i in range(9):
        ok &= np.array([len(set(x)) == 9 for x in g[:, i, :]]) & np.array([len(set(x)) == 9 for x in g[:, :, i]])
    for br in range(3):
        for bc in range(3):
            ok &= np.array([len(set(x.flatten())) == 9 for x in g[:, br*3:br*3+3, bc*3:bc*3+3]])
    return ok


assert valid(augment(train[:64])).all(), "augmentation broke validity"

vq = VQVAE(**cfg.tokenizer.vqvae).cuda().train()
params = [p for p in vq.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, betas=(0.9, 0.99), weight_decay=0.01)
WARM, PEAK, FLOOR = 500, 5e-4, 5e-5
lr_at = lambda s: PEAK * s / WARM if s < WARM else FLOOR + 0.5 * (PEAK - FLOOR) * (1 + math.cos(math.pi * (s - WARM) / max(a.steps - WARM, 1)))


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    vqm, levels = model.vq, list(model.vq.levels)
    hit = np.zeros(len(levels)); ex = np.zeros(len(levels)); seen = [set() for _ in levels]
    for s0 in range(0, len(data), 250):  # chunks of 250 boards keep eval memory small
        x = torch.tensor(data[s0:s0 + 250], device="cuda")
        idx, zh = vqm.z_to_idx(model.encode_z(x), return_zhats=True)
        for k, z in enumerate(zh):
            y = model.decoder(z).argmax(-1)
            hit[k] += float((y == x).float().sum()); ex[k] += float((y == x).all(1).sum())
        a_ = 0
        for k, l in enumerate(levels):
            seen[k].update(idx[:, a_:a_ + l].flatten().tolist()); a_ += l
    acc = [100 * h / (len(data) * 89) for h in hit]
    exact = [100 * e / len(data) for e in ex]
    used = [len(x) for x in seen]
    model.train()
    V = list(cfg.tokenizer.vqvae.vqvae_vocab_size)
    return {"acc_pct": acc, "seq_exact_pct": exact, "used_codes": used, "util_mean_pct": 100 * float(np.mean([u / v for u, v in zip(used, V)]))}


log = {"args": vars(a), "env": env_info(), "n_train": len(train), "n_test": len(test), "evals": []}
print(f"train {len(train)} boards, test {len(test)} boards, params {sum(p.numel() for p in vq.parameters())/1e6:.2f}M, "
      f"fine_drop {cfg.tokenizer.vqvae.vector_quantizer_config.aux.fine_drop.prob}, aug {a.aug}")
t0 = time.time()
for step in range(a.steps + 1):
    if step % a.eval_every == 0 or step == a.steps:
        ev = evaluate(vq, test); ev["step"] = step; ev["elapsed_s"] = time.time() - t0
        ev["train_subset"] = evaluate(vq, train[:1000])["acc_pct"][-1]
        log["evals"].append(ev)
        print(f"[eval] step {step} {ev['elapsed_s']:.0f}s held-out acc final {ev['acc_pct'][-1]:.3f}% exact {ev['seq_exact_pct'][-1]:.1f}% "
              f"util {ev['util_mean_pct']:.2f}% train-acc {ev['train_subset']:.3f}% per-level acc {[round(v, 1) for v in ev['acc_pct']]}", flush=True)
        if step == a.steps:
            break
    xb = train[np.random.randint(0, len(train), a.bs)]
    if a.aug:
        xb = augment(xb)
    for gpar in opt.param_groups:
        gpar["lr"] = lr_at(step)
    opt.zero_grad(set_to_none=True)
    mb = a.bs // a.accum
    for j in range(a.accum):  # same gradient as one batch of a.bs (mean over micro-batches); EMA codebooks update per micro-batch
        x = torch.tensor(xb[j * mb:(j + 1) * mb], device="cuda")
        logits, vq_loss, util, gutil, _ = vq(x)
        rec = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
        loss = rec + vq_loss
        (loss / a.accum).backward()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if step % 100 == 0:
        print(f"step {step:5d} lr {lr_at(step):.2e} rec {rec.item():.4f} vq {float(vq_loss):.4f} gnorm {float(gn):.2f} "
              f"hit-util {100*gutil:.1f}% {time.time()-t0:.0f}s", flush=True)
log["train_seconds"] = time.time() - t0
# released tokenizer on the same held-out boards
_, tk, _, _ = load(cfgpath("sudoku-hard"), os.path.join(ART, "sudoku", "hard"), generator=False)
log["released_hard_tokenizer_on_heldout"] = evaluate(tk.vqvae, test)
tk.vqvae.eval()
print("released sudoku-hard tokenizer on same held-out boards:", log["released_hard_tokenizer_on_heldout"]["acc_pct"][-1],
      "util", log["released_hard_tokenizer_on_heldout"]["util_mean_pct"])
dump(os.path.join(OUT, f"train_{a.tag}.json"), log)
torch.save({"model": vq.state_dict()}, os.path.join(ART, f"p10_tokenizer_{a.tag}.pt"))

"""Shared helpers: load() and partial_decodes() copied from docs/code/guides/use-ncm-from-python.md."""
import os, sys, json, time
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NCM = os.path.abspath(os.path.join(EXP, "..", "ncm-main"))
ART = os.path.join(EXP, "artifacts")
DATA = os.path.join(EXP, "data")
sys.path.insert(0, NCM)
import torch
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

# Cap the caching allocator (default 0.88 of 4 GB): on Windows an overflow spills to shared memory and stalls
# instead of raising OOM. With the cap, an oversize batch raises torch.cuda.OutOfMemoryError.
MEM_FRACTION = float(os.environ.get("NCM_MEM_FRACTION", "0.88"))
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)


def cfgpath(name):
    return os.path.join(NCM, "config", "eval", f"{name}.yaml")


def load(config_path, ck, generator=True):
    cfg = OmegaConf.load(config_path)
    cfg.tokenizer.vqvae.checkpoint_path = os.path.join(ck, cfg.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            cfg.tokenizer.pre_tokenizer[key] = os.path.join(ck, cfg.checkpoints[name])
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae = tk.vqvae.cuda().eval()
    if not generator:
        return cfg, tk, None, None
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).cuda()
    sd = torch.load(os.path.join(ck, cfg.checkpoints.generator), map_location="cpu", weights_only=False)
    net.load_state_dict(sd["model"], strict=True)
    return cfg, tk, net.eval(), sd


@torch.no_grad()
def partial_decodes(vqvae, codes, logits=False):
    vq = vqvae.vq
    L, K = vq.levels[-1], len(vq.levels)
    z_hat = torch.zeros(codes.shape[0], L, vq.embed_dim, device=codes.device)
    out, a = [], 0
    for k, l in enumerate(vq.levels):
        e = vq.embeddings[k](codes[:, a:a + l])
        q = vq._up_to_full(e, k, L) if k < K - 1 else e
        z_hat = z_hat + vq._apply_quant_resi(q, k)
        d = vqvae.decoder(z_hat)
        out.append(d if logits else d.argmax(-1))
        a += l
    return out


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def env_info():
    return {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mem_fraction_cap": MEM_FRACTION}

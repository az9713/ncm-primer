"""Project 9 follow-up 2: which part of the timing loop holds GPU memory? Batch 16, config sampler, 20 calls per variant:
(a) generate only, no gc.collect(); (b) generate + idx_to_zhats + decoder argmax (chunks of 16), with gc.collect();
(c) the same as (b) without gc.collect(). memory_allocated after each call."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
cfg, tk, net, sd = load(cfgpath("lm1b-small"), os.path.join(ART, "lm1b", "small"))
EOS = tk.pre_tokenizer.eos_token_id
p = sd["inference"]["level0_unigram"].float().cpu()
G = cfg.generation
B = 16
kw = dict(temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p))
res = {"env": env_info(), "B": B, "variants": {}}


def one(decode):
    l1 = torch.multinomial(p, B, replacement=True).view(B, 1).cuda()
    g = net.generate(l1, torch.full((B, 1), EOS, device="cuda"), **kw)
    if not decode:
        return g
    _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
    return torch.cat([tk.vqvae.decoder(zl[-1][s:s + 16]).argmax(-1) for s in range(0, B, 16)])


for name, decode, collect in (("a_generate_no_gc", False, False), ("b_decode_gc", True, True), ("c_decode_no_gc", True, False)):
    vals = []
    for i in range(20):
        try:
            with torch.no_grad():
                out = one(decode)
            del out
            if collect:
                gc.collect()
            vals.append(torch.cuda.memory_allocated() / 2**20)
        except torch.cuda.OutOfMemoryError as e:
            vals.append(f"OOM: {str(e)[:120]}"); break
    res["variants"][name] = vals
    print(f"{name:18s} allocated MiB per call: {[round(v) if isinstance(v, float) else v for v in vals]}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
dump(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leak_test2.json"), res)

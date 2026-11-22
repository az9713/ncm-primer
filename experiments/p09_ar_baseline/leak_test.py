"""Project 9 follow-up: does live GPU memory grow per NCM LM1B generate() call? 20 calls at batch 16, config sampler,
memory_allocated after each call (no decode), then 10 calls with argmax sampling (temperature 0) for comparison."""
import sys, os, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
cfg, tk, net, sd = load(cfgpath("lm1b-small"), os.path.join(ART, "lm1b", "small"))
EOS = tk.pre_tokenizer.eos_token_id
p = sd["inference"]["level0_unigram"].float().cpu()
G = cfg.generation
B = 16
res = {"env": env_info(), "B": B, "after_load_MiB": torch.cuda.memory_allocated() / 2**20, "calls": []}
print(f"after load: {res['after_load_MiB']:.0f} MiB allocated", flush=True)
for mode, kw, n in (("config sampler (T=1, top_k=0, top_p=1)", dict(temperature=float(G.temperature), top_k=int(G.top_k), top_p=float(G.top_p)), 20),
                    ("argmax (T=0)", dict(temperature=0.0, top_k=0, top_p=1.0), 10)):
    for i in range(n):
        try:
            with torch.no_grad():
                l1 = torch.multinomial(p, B, replacement=True).view(B, 1).cuda()
                g = net.generate(l1, torch.full((B, 1), EOS, device="cuda"), **kw)
            del g, l1; gc.collect()
            a = torch.cuda.memory_allocated() / 2**20
            res["calls"].append({"mode": mode, "i": i, "allocated_MiB": a})
            print(f"{mode:40s} call {i:2d}: allocated {a:.0f} MiB", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            res["calls"].append({"mode": mode, "i": i, "error": str(e)[:200]}); print(f"{mode} call {i}: OOM"); break
    torch.cuda.empty_cache()
dump(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leak_test.json"), res)

"""Project 5 follow-up: WMT14 (and XSum) tokenizer accuracy restricted to references whose tokens + EOS fit in L,
so no row is truncated. Mask = text tokens + first EOS; rows padded with PAD."""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
from transformers import AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
res = {"env": env_info()}
for task, f, bs in (("wmt14", "wmt14/reference_translations.pickle", 16), ("xsum", "xsum/reference_summaries.pickle", 32)):
    cfg, tk, _, _ = load(cfgpath(task), os.path.join(ART, task), generator=False)
    L, EOS, PAD = tk.vqvae.context_length, tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    st = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
    ids = [st(s)["input_ids"] for s in pickle.load(open(os.path.join(DATA, f), "rb"))]
    for name, sel in (("all", ids), ("fits_in_L", [x for x in ids if len(x) + 1 <= L]), ("truncated", [x for x in ids if len(x) + 1 > L])):
        if not sel:
            continue
        hit = tot = ex = 0
        for s0 in range(0, len(sel), bs):
            rows = [((x + [EOS])[:L] + [PAD] * L)[:L] for x in sel[s0:s0 + bs]]
            m = torch.tensor([[1] * min(len(x) + 1, L) + [0] * (L - min(len(x) + 1, L)) for x in sel[s0:s0 + bs]], device="cuda").bool()
            x = torch.tensor(rows, device="cuda")
            with torch.no_grad():
                y = tk.vqvae.decoder(tk.vqvae.vq.z_to_idx(tk.vqvae.encode_z(x), return_zhats=True)[1][-1]).argmax(-1)
            eq = y == x
            hit += (eq & m).sum().item(); tot += m.sum().item(); ex += (eq | ~m).all(1).sum().item()
        res.setdefault(task, {})[name] = {"rows": len(sel), "text_acc_pct": 100 * hit / tot, "rows_exact_pct": 100 * ex / len(sel)}
        print(f"{task:6s} {name:10s} rows {len(sel):5d}  text+EOS acc {100*hit/tot:.3f}%  rows exact {100*ex/len(sel):.2f}%")
    del tk; torch.cuda.empty_cache()
dump(os.path.join(OUT, "wmt_fit_check.json"), res)

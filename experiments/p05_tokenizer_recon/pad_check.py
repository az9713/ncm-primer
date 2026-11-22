"""Project 5 follow-up: what does the decoder output at pad positions (XSum, WMT14, TinyStories), and does an EOS-filled
row format change the non-pad accuracy? Mask for 'text' accuracy = text tokens + the first EOS (same mask in both formats)."""
import sys, os, pickle, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from ncm_helpers import *
from transformers import AutoTokenizer

OUT = os.path.dirname(os.path.abspath(__file__))
res = {"env": env_info()}
for task, ck, bs in (("xsum", "xsum", 32), ("wmt14", "wmt14", 16), ("tinystories", "tinystories", 64)):
    cfg, tk, _, _ = load(cfgpath(task), os.path.join(ART, ck), generator=False)
    L, EOS, PAD = tk.vqvae.context_length, tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
    if task == "tinystories":
        texts = pickle.load(open(os.path.join(DATA, "tinystories", "valid_bpe_8192.pickle"), "rb"))
    else:
        f = {"xsum": "xsum/reference_summaries.pickle", "wmt14": "wmt14/reference_translations.pickle"}[task]
        st = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
        texts = [st(s)["input_ids"] for s in pickle.load(open(os.path.join(DATA, f), "rb"))]
    out = {}
    for fmt, filler in (("eos_then_pad", PAD), ("eos_fill", EOS)):
        rows, masks = [], []
        for ids in texts:
            t = (list(ids) + [EOS])[:L]
            rows.append((t + [filler] * L)[:L]); masks.append([1] * len(t) + [0] * (L - len(t)))
        hit = tot = phit = ptot = 0; pred_at_pad = collections.Counter(); exact = 0
        for s in range(0, len(rows), bs):
            x = torch.tensor(rows[s:s + bs], device="cuda"); m = torch.tensor(masks[s:s + bs], device="cuda").bool()
            with torch.no_grad():
                y = tk.vqvae.decoder(tk.vqvae.vq.z_to_idx(tk.vqvae.encode_z(x), return_zhats=True)[1][-1]).argmax(-1)
            eq = y == x
            hit += (eq & m).sum().item(); tot += m.sum().item(); phit += (eq & ~m).sum().item(); ptot += (~m).sum().item()
            exact += ((eq | ~m).all(1)).sum().item()
            pred_at_pad.update(y[~m].tolist())
        top = [(int(k), v / max(ptot, 1)) for k, v in pred_at_pad.most_common(3)]
        out[fmt] = {"text_acc_pct": 100 * hit / tot, "fill_positions_acc_pct": 100 * phit / max(ptot, 1), "fill_share_pct": 100 * ptot / (tot + ptot),
                    "rows_exact_on_text_pct": 100 * exact / len(rows), "top_pred_at_fill_positions": top,
                    "top_pred_tokens": [tk.pre_tokenizer.decode([k]) if k is not None else None for k, _ in top]}
        print(f"{task:11s} {fmt:12s} text+EOS acc {out[fmt]['text_acc_pct']:.3f}%  fill-position acc {out[fmt]['fill_positions_acc_pct']:.2f}% "
              f"(fill share {out[fmt]['fill_share_pct']:.1f}%)  rows exact on text {out[fmt]['rows_exact_on_text_pct']:.2f}%  "
              f"top predictions at fill positions {[(k, round(v, 3)) for k, v in top]} {out[fmt]['top_pred_tokens']}  (EOS={EOS}, PAD={PAD})")
    res[task] = out
    del tk; torch.cuda.empty_cache()
dump(os.path.join(OUT, "pad_check.json"), res)

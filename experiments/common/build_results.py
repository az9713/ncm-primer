"""Build results.json for projects 1, 2, 3 by parsing their run logs (numbers are never typed by hand)."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from parse_logs import parse

E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def runs(*logs):
    out = []
    for l in logs:
        p = os.path.join(E, l)
        if os.path.exists(p):
            for r in parse(p):
                r["log"] = l
                out.append(r)
    return out


def label(r):
    o = r.get("out") or ""
    if not o and "--glob" in r["cmd"]:
        o = r["cmd"].split("--glob ")[1].split()[0]
    return os.path.basename(os.path.dirname(o)) if o.endswith(".pkl") else os.path.basename(o)


def scored(rs):
    """Keep blocks that carry metrics; key by arm label (the --out folder or the glob folder)."""
    d = {}
    for r in rs:
        if any(k in r for k in ("genppl_mean", "bleu", "rougeL", "solved_pct")):
            d.setdefault(label(r) + ("|" + ("cpu_fp32" if "--device cpu" in r["cmd"] else "fp16" if "--dtype fp16" in r["cmd"] else "official")), r)
    return d


# Project 1
p1 = {"paper": {"lm1b_table8_p26": {"genppl": 93.05, "entropy": 4.31}, "lm1b_data_table8": {"genppl": 53.51, "entropy": 4.31},
                "xsum_table2_p10": {"rouge1": 35.0, "rouge2": 12.8, "rougeL": 28.0}},
      "runs": scored(runs("p01_penalties/run.log", "p01_penalties/run_arms.log", "p01_penalties/score_validation.log", "p01_penalties/run_xsum.log")),
      "all_blocks": runs("p01_penalties/run.log", "p01_penalties/run_arms.log", "p01_penalties/score_validation.log", "p01_penalties/run_xsum.log")}
json.dump(p1, open(os.path.join(E, "p01_penalties", "results.json"), "w"), indent=2)
# Project 2
p2 = {"paper": {"table2_p10": {"wmt14_ncm": 22.6, "wmt14_cfg": 23.9, "wmt14_cfg_nfe": 22, "xsum_ncm": [35.0, 12.8, 28.0], "xsum_cfg": [35.3, 13.0, 28.2]},
                "fig14_p30": "ROUGE peaks at w=0.5 (Table 2 setting); BLEU peaks at w=1.0"},
      "runs": scored(runs("p02_guidance_sweep/run_wmt_cfg.log", "p02_guidance_sweep/run_wmt_cfg1k.log", "p02_guidance_sweep/run_wmt_sampler.log", "p02_guidance_sweep/run_xsum_cfg.log", "p01_penalties/run_xsum.log")),
      "all_blocks": runs("p02_guidance_sweep/run_wmt_cfg.log", "p02_guidance_sweep/run_wmt_cfg1k.log", "p02_guidance_sweep/run_wmt_sampler.log", "p02_guidance_sweep/run_xsum_cfg.log")}
json.dump(p2, open(os.path.join(E, "p02_guidance_sweep", "results.json"), "w"), indent=2)
# Project 3
pr = os.path.join(E, "p03_level1_probe", "probe_results.json")
p3 = {"paper": {"tinystories_table8_p26": {"genppl": 15.41, "entropy": 3.98}, "data": {"genppl": 9.67, "entropy": 3.96}},
      "runs": scored(runs("p03_level1_probe/run.log", "p03_level1_probe/score_validation.log")),
      "probe": json.load(open(pr)) if os.path.exists(pr) else None}
json.dump(p3, open(os.path.join(E, "p03_level1_probe", "results.json"), "w"), indent=2)
for name, d in (("p1", p1), ("p2", p2), ("p3", p3)):
    print(name)
    for k, r in d["runs"].items():
        print("  ", k, {x: r[x] for x in ("genppl_mean", "genppl_std", "entropy_mean", "bleu", "chrf", "len_ratio", "rouge1", "rouge2", "rougeL", "genppl_gpt2", "words", "duration_s") if x in r})

# Project 3: add the seed-control analysis
se = os.path.join(E, "p03_level1_probe", "seed_effect.json")
if os.path.exists(se):
    p3["seed_effect"] = json.load(open(se))
    json.dump(p3, open(os.path.join(E, "p03_level1_probe", "results.json"), "w"), indent=2)
# Projects 8, 9, 10: one results.json that collects the per-run JSON files written by the scripts
for proj, pats in (("p08_mbr", ["results_n*.json", "lc_bleu.json"]),
                   ("p09_ar_baseline", ["results_run1_B1only.json", "results_small_batches.json", "results_b4_fresh.json", "results_b8_fresh.json", "results_b16_fresh.json", "results_short_runs.json", "results_gcfix.json", "leak_test.json", "leak_test2.json", "results_run2.json"]),
                   ("p10_train_tokenizer", ["train_real*.json", "train_aug*.json", "train_timing100.json"])):
    agg = {}
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(E, proj, pat))):
            agg[os.path.basename(f)] = json.load(open(f))
    if proj == "p09_ar_baseline" and os.path.exists(os.path.join(E, proj, "results.json")) and "results_run2.json" not in agg:
        r2 = json.load(open(os.path.join(E, proj, "results.json")))
        if "runs" not in r2:  # results.json of run 2 (phase-separated script)
            agg["results_run2.json"] = r2
    out = os.path.join(E, proj, "results.json" if proj != "p09_ar_baseline" else "results_all.json")
    json.dump(agg, open(out, "w"), indent=2)
    print(proj, "->", os.path.basename(out), list(agg))

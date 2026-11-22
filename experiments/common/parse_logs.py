"""Parse evaluate.py merge output blocks from a run.log produced by common/run.sh into a list of dicts."""
import re, sys, json


def parse(path):
    txt = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for blk in txt.split("### CMD: ")[1:]:
        cmd = blk.splitlines()[0]
        d = {"cmd": cmd}
        m = re.search(r"### START: (\S+)", blk); d["start"] = m.group(1) if m else None
        m = re.search(r"### END: (\S+) rc=(\d+) duration_s=(\d+)", blk)
        if m: d["end"], d["rc"], d["duration_s"] = m.group(1), int(m.group(2)), int(m.group(3))
        m = re.search(r"--out (\S+)", cmd); d["out"] = m.group(1) if m else None
        m = re.search(r"merged \d+ shards -> (\d+)", blk); d["n_outputs"] = int(m.group(1)) if m else None
        m = re.search(r"solved %.*\n\s*([\d.]+)\s+(\d+)/(\d+)\s+([\d.]+)\s+([\d.]+)", blk)
        if m: d.update(solved_pct=float(m.group(1)), solved=int(m.group(2)), n=int(m.group(3)), cell_pct=float(m.group(4)), valid_pct=float(m.group(5)))
        m = re.search(r"BLEU\s+BLEU\(lc\)\s+chrF\s+len ratio\s*\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", blk)
        if m: d.update(bleu=float(m.group(1)), bleu_lc=float(m.group(2)), chrf=float(m.group(3)), len_ratio=float(m.group(4)))
        m = re.search(r"R-1\s+R-2\s+R-L\s+genPPL\s+words\s*\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", blk)
        if m: d.update(rouge1=float(m.group(1)), rouge2=float(m.group(2)), rougeL=float(m.group(3)), genppl_gpt2=float(m.group(4)), words=float(m.group(5)))
        seeds = re.findall(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", blk, re.M)
        m = re.search(r"mean\s+([\d.]+)\s+([\d.]+)\s+\(\+/- ([\d.]+) over (\d+) seeds, (\d+) samples\)", blk)
        if m:
            d.update(genppl_mean=float(m.group(1)), entropy_mean=float(m.group(2)), genppl_std=float(m.group(3)), n_seeds=int(m.group(4)), n_samples=int(m.group(5)))
            d["per_seed"] = [{"seed": int(a), "genppl": float(b), "entropy": float(c)} for a, b, c in seeds]
        out.append(d)
    return out


if __name__ == "__main__":
    print(json.dumps(parse(sys.argv[1]), indent=2))

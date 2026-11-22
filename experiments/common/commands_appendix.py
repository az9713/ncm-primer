"""Write report_parts/commands.md: every '### CMD:' line from the project logs, with log file, start time, exit code, duration."""
import os, re, glob
E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
out = ["## 5. Appendix: exact command lines", "",
       "Every job was started by `experiments/common/run.sh`, which wrote these lines into the logs. Paths are as logged. Jobs that a stale queue skipped (`common/skipped.log`) are not listed. rc = exit code.", ""]
for d in sorted(glob.glob(os.path.join(E, "p[0-9][0-9]_*"))):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        if os.path.basename(f) == "run_all.log":
            continue
        txt = open(f, encoding="utf-8", errors="replace").read()
        if txt.startswith("# Combined log"):
            continue  # generated concatenation of the other logs in the folder
        for blk in txt.split("### CMD: ")[1:]:
            cmd = blk.splitlines()[0].strip()
            st = re.search(r"### START: (\S+)", blk); en = re.search(r"### END: \S+ rc=(\d+) duration_s=(\d+)", blk)
            rows.append((st.group(1) if st else "", os.path.basename(f), cmd, en.group(1) if en else "?", en.group(2) if en else "?"))
    if not rows:
        continue
    out += [f"### {os.path.basename(d)}", ""]
    for st, f, cmd, rc, du in sorted(rows):
        out.append(f"- `{f}` {st} rc={rc} {du} s:")
        out.append(f"  `{cmd}`")
    out.append("")
open(os.path.join(E, "report_parts", "commands.md"), "w", encoding="utf-8").write("\n".join(out))
print("commands.md lines:", len(out))

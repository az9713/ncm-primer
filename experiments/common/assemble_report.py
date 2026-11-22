"""Concatenate report_parts into REPORT.md, then convert to REPORT.html. Fails if a placeholder is left."""
import os, re, subprocess, sys
E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
subprocess.run([sys.executable, os.path.join(E, "common", "commands_appendix.py")], check=True)
parts = ["header", "env"] + [f"p{n:02d}" for n in range(1, 11)] + ["cross", "commands"]
order = {"header": "", "env": "", "cross": ""}
body = []
for p in parts:
    txt = open(os.path.join(E, "report_parts", p + ".md"), encoding="utf-8").read().strip()
    if p.startswith("p"):
        txt = txt.replace("## Project", "### Project", 1)
    body.append(txt)
md = "\n\n".join(body[:2]) + "\n\n## 3. Projects\n\n" + "\n\n".join(body[2:-1]) + "\n\n" + body[-1] + "\n"
left = sorted(set(re.findall(r"[A-Z0-9_]*PLACEHOLDER[A-Z0-9_]*|P9_(?:STATUS|RESULT)|P10_(?:STATUS|RESULT|CONSISTENT)|FRESH_[A-Z0-9_]+", md)))
open(os.path.join(E, "REPORT.md"), "w", encoding="utf-8").write(md)
print("wrote REPORT.md", len(md), "chars; placeholders left:", left)
if left and "--draft" not in sys.argv:
    sys.exit(f"placeholders left: {left}")
subprocess.run([sys.executable, os.path.join(E, "common", "md2html.py"), os.path.join(E, "REPORT.md"), os.path.join(E, "REPORT.html")], check=True)

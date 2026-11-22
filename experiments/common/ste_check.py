"""List sentences longer than 25 words in report_parts/*.md (tables, code blocks, headings and the appendix are skipped)."""
import os, re, glob, sys
E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIM = int(sys.argv[1]) if len(sys.argv) > 1 else 25
tot = 0
for f in sorted(glob.glob(os.path.join(E, "report_parts", "*.md"))):
    if f.endswith("commands.md"):
        continue
    incode = False
    for n, line in enumerate(open(f, encoding="utf-8"), 1):
        if line.startswith("```"):
            incode = not incode; continue
        if incode or line.lstrip().startswith("|") or line.startswith("#") or not line.strip():
            continue
        text = re.sub(r"`[^`]*`", "X", line)            # inline code counts as one word
        text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
        for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z(“\"])", text.strip().lstrip("-0123456789. ")):
            w = len(sent.split())
            if w > LIM:
                tot += 1
                print(f"{os.path.basename(f)}:{n} [{w}] {sent[:150]}")
print("sentences over", LIM, "words:", tot)

"""Compare numeric tokens and inline-code spans of report_parts/<f> against report_parts_orig/<f>. Prints differences."""
import os, re, sys, glob, collections
E = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
files = sys.argv[1:] or [os.path.basename(f) for f in glob.glob(os.path.join(E, "report_parts", "*.md"))]
bad = 0
for f in files:
    a = open(os.path.join(E, "report_parts_orig", f), encoding="utf-8").read()
    b = open(os.path.join(E, "report_parts", f), encoding="utf-8").read()
    num = lambda s: collections.Counter(re.findall(r"\d[\d,]*(?:\.\d+)?", s))
    code = lambda s: collections.Counter(re.findall(r"`[^`]+`", s))
    for name, fa, fb in (("numbers", num(a), num(b)), ("code spans", code(a), code(b))):
        lost = fa - fb; added = fb - fa
        if lost or added:
            bad += 1
            print(f"{f}: {name} lost {dict(lost)} added {dict(added)}")
print("files with differences:", bad)

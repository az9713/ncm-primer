"""Convert experiments/REPORT.md to REPORT.html (dark mode). Converts only the one file named on the command line."""
import sys, markdown, html
src, dst = sys.argv[1], sys.argv[2]
body = markdown.markdown(open(src, encoding="utf-8").read(), extensions=["tables", "fenced_code", "toc", "sane_lists"])
css = """:root{color-scheme:dark}body{background:#0f172a;color:#e2e8f0;font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:16px}
main{max-width:1100px;margin:0 auto}h1,h2,h3{color:#fb923c;line-height:1.25}h2{border-bottom:1px solid #334155;padding-bottom:4px;margin-top:2em}
h3{color:#2dd4bf}a{color:#2dd4bf}p,li{color:#cbd5e1}code{background:#1e293b;color:#2dd4bf;padding:1px 4px;border-radius:4px;font-size:.92em;overflow-wrap:anywhere}
pre{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:12px;overflow-x:auto}pre code{padding:0;background:none;color:#e2e8f0}
table{border-collapse:collapse;display:block;overflow-x:auto;margin:12px 0}th,td{border:1px solid #334155;padding:5px 8px;vertical-align:top}
th{background:#1e293b;color:#e2e8f0}td{color:#cbd5e1}blockquote{border-left:3px solid #fb923c;margin:0;padding:4px 12px;background:#1e293b}
.muted{color:#94a3b8}"""
page = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<meta name=color-scheme content=dark><title>NCM Experiment Report</title><style>{css}</style></head><body><main>{body}</main></body></html>")
open(dst, "w", encoding="utf-8").write(page)
print("wrote", dst)

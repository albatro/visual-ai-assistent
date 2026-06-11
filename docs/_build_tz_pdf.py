import markdown
import subprocess
from pathlib import Path

DOCS = Path(__file__).parent
md_path = DOCS / "ТЗ.md"
html_path = DOCS / "_tz.html"
pdf_path = DOCS / "ТЗ.pdf"
edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

md_text = md_path.read_text(encoding="utf-8")

html_body = markdown.markdown(
    md_text,
    extensions=["extra", "tables", "sane_lists", "toc"],
)

css = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', 'Calibri', 'Tahoma', 'Arial Unicode MS', Arial, sans-serif;
  font-size: 10.5pt;
  line-height: 1.45;
  color: #1f2328;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
h1 {
  font-size: 20pt;
  border-bottom: 2px solid #1f2328;
  padding-bottom: 6px;
  margin: 0 0 14px 0;
  page-break-after: avoid;
}
h2 {
  font-size: 14pt;
  border-bottom: 1px solid #d0d7de;
  padding-bottom: 3px;
  margin: 20px 0 8px 0;
  page-break-after: avoid;
}
h3 {
  font-size: 12pt;
  margin: 14px 0 6px 0;
  page-break-after: avoid;
}
p, ul, ol { margin: 5px 0 9px 0; }
ul, ol { padding-left: 22px; }
li { margin: 2px 0; }
strong { color: #0a0a0a; }
em { color: #333; }
code {
  font-family: 'Consolas', 'Courier New', monospace;
  background: #f3f4f6;
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 9.5pt;
}
blockquote {
  border-left: 3px solid #d0d7de;
  margin: 10px 0;
  padding: 6px 12px;
  color: #444;
  background: #f8fafc;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 8px 0 12px 0;
  font-size: 9.5pt;
  page-break-inside: auto;
  table-layout: fixed;
  word-wrap: break-word;
}
th, td {
  border: 1px solid #c8d0d8;
  padding: 5px 8px;
  text-align: left;
  vertical-align: top;
}
th {
  background: #eef1f5;
  font-weight: 600;
}
hr {
  border: none;
  border-top: 1px solid #d0d7de;
  margin: 18px 0;
}
"""

html_full = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>ТЗ</title>
<style>{css}</style>
</head>
<body>
{html_body}
</body>
</html>
"""

html_path.write_text(html_full, encoding="utf-8")
print(f"HTML written: {html_path}")

# Render PDF via headless Edge with no browser-added headers/footers.
url = html_path.absolute().as_uri()
cmd = [
    edge_exe,
    "--headless=new",
    "--disable-gpu",
    "--no-pdf-header-footer",
    "--run-all-compositor-stages-before-draw",
    "--virtual-time-budget=10000",
    f"--print-to-pdf={pdf_path}",
    url,
]
print("Running Edge:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    raise SystemExit(f"Edge failed with code {result.returncode}")
print(f"PDF written: {pdf_path} ({pdf_path.stat().st_size} bytes)")

# Cleanup intermediate HTML
try:
    html_path.unlink()
except OSError:
    pass

"""docs/report.md を論文風の PDF にする。

    python tools/build_report.py --account <omnicampus のアカウント名>

出力: docs/説明資料_PAI最終課題_<account>.pdf
"""

import argparse
from pathlib import Path

import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm;
        @bottom-center { content: counter(page); font-size: 9pt; color: #555; } }
body { font-family: "Noto Serif CJK JP", serif; font-size: 10pt; line-height: 1.65; color: #111; }
h1 { font-family: "Noto Sans CJK JP", sans-serif; font-size: 16pt; text-align: center; line-height: 1.4;
     margin: 0 0 14pt 0; }
h2 { font-family: "Noto Sans CJK JP", sans-serif; font-size: 12pt; border-bottom: 1px solid #333;
     margin: 16pt 0 6pt 0; padding-bottom: 2pt; }
h3 { font-family: "Noto Sans CJK JP", sans-serif; font-size: 10.5pt; margin: 10pt 0 4pt 0; }
p { margin: 0 0 6pt 0; text-align: justify; text-indent: 1em; }
li p, .abstract p, .refs p { text-indent: 0; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 1.6em; }
li { margin-bottom: 2pt; }
.abstract { border: 1px solid #999; background: #f6f6f6; padding: 8pt 12pt; margin: 0 0 10pt 0;
            font-size: 9.5pt; }
.abstract p { margin-bottom: 4pt; }
figure { margin: 8pt 0 10pt 0; text-align: center; page-break-inside: avoid; }
figure img { width: 100%; }
figure img.half { width: 72%; }
figcaption { font-size: 9pt; margin-top: 3pt; }
.caption { font-size: 9pt; text-align: center; margin: 8pt 0 2pt 0; }
table { border-collapse: collapse; margin: 2pt auto 10pt auto; font-size: 8.8pt; page-break-inside: avoid; }
th, td { border-top: 1px solid #444; border-bottom: 1px solid #444; padding: 2.5pt 7pt; }
th { background: #eee; }
pre { font-size: 7.6pt; background: #f4f4f4; padding: 6pt; white-space: pre-wrap; line-height: 1.4; }
code { font-family: "Noto Sans Mono CJK JP", monospace; font-size: 8.5pt; }
.refs { font-size: 8.6pt; line-height: 1.45; }
.refs li { margin-bottom: 3pt; }
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="アカウント名")
    args = ap.parse_args()
    src = (ROOT / "docs" / "report.md").read_text(encoding="utf-8")
    body = markdown.markdown(src, extensions=["tables", "fenced_code", "md_in_html"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    out = ROOT / "docs" / f"説明資料_PAI最終課題_{args.account}.pdf"
    HTML(string=html, base_url=str(ROOT / "docs")).write_pdf(out)
    print(out)


if __name__ == "__main__":
    main()

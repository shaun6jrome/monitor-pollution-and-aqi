"""Convert Academic Report Markdown into Printable HTML and PDF.

Reads docs/report.md, formats it with professional typography and page margins,
and uses headless Google Chrome to compile docs/report.pdf.
"""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def markdown_to_html_simple(md_text: str) -> str:
    """Convert basic markdown elements (headers, tables, code blocks, lists) to clean HTML."""
    lines = md_text.splitlines()
    html_lines = []
    in_code = False
    in_table = False

    for line in lines:
        # Code fences
        if line.strip().startswith("```"):
            if in_code:
                html_lines.append("</pre>")
                in_code = False
            else:
                lang = line.strip().replace("```", "")
                html_lines.append(f"<pre class='code-block {lang}'>")
                in_code = True
            continue

        if in_code:
            html_lines.append(html.escape(line))
            continue

        # Tables
        if "|" in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if not parts:
                continue
            if all(re.match(r"^:?-+:?$", p) for p in parts):
                continue  # Header divider
            if not in_table:
                html_lines.append("<table>")
                html_lines.append("<tr>" + "".join(f"<th>{html.escape(p)}</th>" for p in parts) + "</tr>")
                in_table = True
            else:
                html_lines.append("<tr>" + "".join(f"<td>{html.escape(p)}</td>" for p in parts) + "</tr>")
            continue
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False

        # Headers
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{html.escape(line[5:])}</h4>")
        elif line.strip().startswith("- "):
            html_lines.append(f"<li>{html.escape(line.strip()[2:])}</li>")
        elif re.match(r"^\d+\.\s+", line.strip()):
            content = re.sub(r"^\d+\.\s+", "", line.strip())
            html_lines.append(f"<li>{html.escape(content)}</li>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        elif line.strip():
            # Paragraph
            text = html.escape(line.strip())
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
            html_lines.append(f"<p>{text}</p>")

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)


def compile_report_pdf() -> bool:
    """Generate HTML and PDF report from docs/report.md."""
    md_file = config.DOCS_DIR / "report.md"
    html_file = config.DOCS_DIR / "report.html"
    pdf_file = config.DOCS_DIR / "report.pdf"

    if not md_file.exists():
        logger.error("Report markdown not found at %s", md_file)
        return False

    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    body_html = markdown_to_html_simple(md_content)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Academic Project Report — Pollution Intelligence</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            line-height: 1.6;
            font-size: 11pt;
            background: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            color: #0f172a;
            font-size: 22pt;
            border-bottom: 2px solid #0f172a;
            padding-bottom: 8px;
            margin-top: 0;
        }}
        h2 {{
            color: #1e293b;
            font-size: 15pt;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 24px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #334155;
            font-size: 12pt;
            margin-top: 16px;
            page-break-after: avoid;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0;
            font-size: 9.5pt;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 6px 10px;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            font-weight: 600;
        }}
        pre.code-block {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 10px;
            font-family: ui-monospace, Menlo, Monaco, monospace;
            font-size: 9pt;
            border-radius: 4px;
            overflow-x: auto;
        }}
        code {{
            background-color: #f1f5f9;
            padding: 2px 4px;
            font-size: 9.5pt;
            border-radius: 3px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
{body_html}
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info("Saved formatted HTML report to %s", html_file)

    # Compile to PDF using headless Chrome if available
    if os.path.exists(CHROME_PATH):
        try:
            cmd = [
                CHROME_PATH,
                "--headless",
                "--disable-gpu",
                f"--print-to-pdf={pdf_file}",
                "--no-pdf-header-footer",
                str(html_file),
            ]
            subprocess.run(cmd, check=True, timeout=30)
            logger.info("Successfully generated PDF report at %s", pdf_file)
            return True
        except Exception as e:
            logger.warning("Headless Chrome PDF generation failed: %s", e)
    else:
        logger.info("Google Chrome binary not found at %s. HTML report is ready for printing.", CHROME_PATH)

    return html_file.exists()


if __name__ == "__main__":
    success = compile_report_pdf()
    sys.exit(0 if success else 1)

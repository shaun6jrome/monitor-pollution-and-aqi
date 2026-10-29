"""Generate Beautiful PDF Guide with Embedded Screenshots and Rubric Breakdown.

Converts docs/comprehensive_project_guide.md into an elegant printable HTML document,
embeds local screenshots of the website and the 40-mark evaluation rubric,
and compiles it into docs/comprehensive_project_guide.pdf via headless Google Chrome.
"""

from __future__ import annotations

import base64
import html
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def get_image_base64(img_path: Path) -> str:
    """Read image and return data URI base64 string."""
    if not img_path.exists():
        return ""
    mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def markdown_to_html(md_text: str) -> str:
    """Convert markdown elements to structured HTML with custom visual callouts."""
    lines = md_text.splitlines()
    html_lines = []
    in_code = False
    in_table = False

    for line in lines:
        # Code block
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
                continue
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

        # Headings
        if line.startswith("# "):
            html_lines.append(f"<h1 class='doc-title'>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{html.escape(line[5:])}</h4>")
        elif line.strip().startswith("- "):
            content = html.escape(line.strip()[2:])
            content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.*?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        elif re.match(r"^\d+\.\s+", line.strip()):
            content = re.sub(r"^\d+\.\s+", "", line.strip())
            content = html.escape(content)
            content = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", content)
            content = re.sub(r"`(.*?)`", r"<code>\1</code>", content)
            html_lines.append(f"<li>{content}</li>")
        elif line.strip() == "---":
            html_lines.append("<hr>")
        elif line.strip():
            text = html.escape(line.strip())
            text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
            text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
            html_lines.append(f"<p>{text}</p>")

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)


def compile_comprehensive_guide_pdf() -> bool:
    """Generate HTML and PDF guide with embedded screenshots."""
    md_file = config.DOCS_DIR / "comprehensive_project_guide.md"
    html_file = config.DOCS_DIR / "comprehensive_project_guide.html"
    pdf_file = config.DOCS_DIR / "comprehensive_project_guide.pdf"

    if not md_file.exists():
        logger.error("Markdown file %s not found.", md_file)
        return False

    with open(md_file, "r", encoding="utf-8") as f:
        md_text = f.read()

    body_html = markdown_to_html(md_text)

    # Base64 encode screenshots
    rubric_img = get_image_base64(config.DOCS_DIR / "screenshots" / "rubrics_table.png")
    event_img = get_image_base64(config.DOCS_DIR / "screenshots" / "website_event_explorer.png")
    news_img = get_image_base64(config.DOCS_DIR / "screenshots" / "website_news_intelligence.png")

    visual_gallery_html = f"""
    <div class="visual-gallery">
        <h2>Visual Interface & Evaluation Rubric Artifacts</h2>
        
        <div class="figure-card">
            <h3>Figure 1: Official Academic Evaluation Rubric (40 Marks Total)</h3>
            <p>The academic evaluation rubric defining all 7 criteria totaling 40 marks, including Online Certification (10M), Machine Learning (7M), Dataset (5M), Real-time Implementation (5M), Innovation (3M), Presentation & Viva (5M), and Report (5M).</p>
            <img src="{rubric_img}" alt="Official Evaluation Rubric (40 Marks)" class="img-rubric">
        </div>

        <div class="figure-card">
            <h3>Figure 2: Duplicate-Resistant Pollution Event Explorer Interface</h3>
            <p>Live screenshot of the running Streamlit platform (Port 8501) showing discovered real-world incidents, the multi-criteria clustering engine, and export controls.</p>
            <img src="{event_img}" alt="Pollution Event Explorer Screenshot" class="img-full">
        </div>

        <div class="figure-card">
            <h3>Figure 3: Unstructured Environmental News Intelligence Interface</h3>
            <p>Live screenshot showing real-time and cached GDELT news ingestion, topic dropdowns, article relevance filtering, and evidence cards.</p>
            <img src="{news_img}" alt="News Intelligence Screenshot" class="img-full">
        </div>
    </div>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Comprehensive Project Walkthrough & Rubrics Guide</title>
    <style>
        @page {{
            size: A4;
            margin: 18mm 14mm 18mm 14mm;
            @bottom-right {{
                content: counter(page);
                font-size: 9pt;
                color: #64748b;
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #0f172a;
            line-height: 1.55;
            font-size: 10.5pt;
            background: #ffffff;
            margin: 0;
            padding: 10px;
        }}
        h1.doc-title {{
            color: #0f172a;
            font-size: 20pt;
            border-bottom: 3px solid #059669;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 12px;
        }}
        h2 {{
            color: #1e293b;
            font-size: 14pt;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 22px;
            margin-bottom: 8px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #059669;
            font-size: 11.5pt;
            margin-top: 14px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 4px;
            margin-bottom: 8px;
        }}
        li {{
            margin-bottom: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }}
        th {{
            background-color: #f1f5f9;
            font-weight: 600;
            color: #0f172a;
        }}
        pre.code-block {{
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            font-family: ui-monospace, Menlo, Monaco, monospace;
            font-size: 8.5pt;
            border-radius: 4px;
            overflow-x: auto;
            margin: 10px 0;
        }}
        code {{
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 1px 4px;
            font-size: 9pt;
            border-radius: 3px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 18px 0;
        }}
        .figure-card {{
            border: 1px solid #e2e8f0;
            background-color: #f8fafc;
            border-radius: 6px;
            padding: 12px;
            margin: 16px 0;
            page-break-inside: avoid;
        }}
        .img-rubric {{
            width: 80%;
            max-width: 650px;
            display: block;
            margin: 10px auto;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
        }}
        .img-full {{
            width: 100%;
            display: block;
            margin: 10px 0;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
{body_html}
<hr>
{visual_gallery_html}
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    logger.info("Saved formatted HTML guide to %s", html_file)

    if os.path.exists(CHROME_PATH):
        try:
            cmd = [
                CHROME_PATH,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--user-data-dir=/tmp/chrome_guide_{os.getpid()}",
                f"--print-to-pdf={pdf_file}",
                "--no-pdf-header-footer",
                str(html_file),
            ]
            subprocess.run(cmd, check=True, timeout=40)
            logger.info("Successfully compiled PDF guide at %s (%d bytes)", pdf_file, pdf_file.stat().st_size)
            return True
        except Exception as e:
            logger.error("Headless Chrome PDF generation failed: %s", e)
    else:
        logger.warning("Google Chrome binary not found at %s.", CHROME_PATH)

    return html_file.exists()


if __name__ == "__main__":
    success = compile_comprehensive_guide_pdf()
    sys.exit(0 if success else 1)

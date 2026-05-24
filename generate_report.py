"""Generate a clean PDF report from the notebook — outputs first, code compact."""
import os
import re
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "dating_app_match_prediction.ipynb")
HTML_PATH = os.path.join(BASE_DIR, "notebooks", "dating_app_match_prediction_report.html")
PDF_PATH = os.path.join(BASE_DIR, "notebooks", "dating_app_match_prediction_report.pdf")

# Step 1: Convert notebook to HTML
print("Converting notebook to HTML...")
result = subprocess.run([
    sys.executable, "-m", "nbconvert",
    "--to", "html",
    NOTEBOOK_PATH,
    "--output", "dating_app_match_prediction_report.html",
], capture_output=True, text=True)
if result.returncode != 0:
    print(f"nbconvert error: {result.stderr}")
    sys.exit(1)

with open(HTML_PATH, "r", encoding="utf-8") as f:
    html = f.read()

# Step 2: Strip all existing styles
html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
html = re.sub(r'<link[^>]*stylesheet[^>]*/?>', '', html)

# Step 3: Remove anchor-link pilcrow symbols
html = re.sub(r'<a class="anchor-link"[^>]*>.*?</a>', '', html)

# Step 4: Post-process tables
def fix_table(match):
    table_html = match.group(0)
    # Remove inline text-align style
    table_html = re.sub(r'<tr style="text-align: right;">', '<tr>', table_html)
    # Remove the second header row (Rank row with empty cells)
    table_html = re.sub(r'<thead>(.*?)</thead>', lambda m: _fix_thead(m), table_html, flags=re.DOTALL)
    # Count columns to determine table class
    col_count = len(re.findall(r'<th', table_html.split('</thead>')[0] if '</thead>' in table_html else table_html[:500]))
    if col_count > 10:
        css_class = 'wide-table'
    elif col_count > 7:
        css_class = 'mid-table'
    else:
        css_class = 'std-table'
    # Rebuild with class
    inner = re.sub(r'<table[^>]*>', f'<table class="{css_class}">', table_html, count=1)
    return '<div class="table-wrap">' + inner + '</div>'

def _fix_thead(m):
    thead = m.group(1)
    rows = re.findall(r'<tr>.*?</tr>', thead, flags=re.DOTALL)
    if len(rows) >= 2:
        return '<thead>' + rows[0] + '</thead>'
    return m.group(0)

html = re.sub(r'<table[^>]*>.*?</table>', lambda m: fix_table(m), html, flags=re.DOTALL)

# Step 4b: Convert index <th> in tbody to <td class="index-cell">
def fix_index_cells(match):
    tbody_html = match.group(0)
    # Replace <th>...</th> inside tbody with styled <td> with inline width
    tbody_html = re.sub(r'<th([^>]*)>', r'<td class="index-cell" style="width:50px;padding:5px 10px;text-align:center;background-color:#eaf2f8;font-weight:bold"\1>', tbody_html)
    tbody_html = re.sub(r'</th>', '</td>', tbody_html)
    # Restore the thead <th> tags (undo the replacement in thead)
    def restore_thead(m):
        return re.sub(r'<td class="index-cell"[^>]*>', '<th', m.group(0)).replace('</td>', '</th>')
    tbody_html = re.sub(r'<thead>.*?</thead>', restore_thead, tbody_html, flags=re.DOTALL)
    return tbody_html

html = re.sub(r'<tbody>.*?</tbody>', fix_index_cells, html, flags=re.DOTALL)

# Step 4c: Add colgroup to give first column more width
def add_colgroup(match):
    table_html = match.group(0)
    first_row_match = re.search(r'<thead>.*?<tr>(.*?)</tr>', table_html, flags=re.DOTALL)
    if first_row_match:
        col_count = len(re.findall(r'<t[hd]', first_row_match.group(1)))
        if col_count > 0:
            colgroup = '<colgroup><col style="width:30%">' + '<col>' * (col_count - 1) + '</colgroup>'
            table_html = re.sub(r'(<table[^>]*>)', r'\1' + colgroup, table_html, count=1)
    return table_html

html = re.sub(r'<table[^>]*>.*?</table>', lambda m: add_colgroup(m), html, flags=re.DOTALL)

# Step 5: Inject clean CSS (NO pseudo-selectors)
report_css = """
<style>
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10px;
    margin: 15px 20px;
    color: #2c3e50;
    line-height: 1.4;
}
h1 {
    font-size: 20px;
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 6px;
    margin-top: 20px;
}
h2 {
    font-size: 16px;
    color: #2980b9;
    margin-top: 16px;
    border-bottom: 1px solid #bdc3c7;
    padding-bottom: 3px;
}
h3 {
    font-size: 13px;
    color: #7f8c8d;
    margin-top: 12px;
}
h4 {
    font-size: 11px;
    color: #95a5a6;
    margin-top: 8px;
}
p { margin: 4px 0; }
ul, ol { margin: 4px 0; padding-left: 20px; }
li { margin: 2px 0; }
strong { color: #2c3e50; }
pre {
    background: #f4f4f4;
    padding: 6px 8px;
    border: 1px solid #ddd;
    border-radius: 3px;
    font-size: 8px;
    font-family: Courier, monospace;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 3px 0;
}
code {
    font-family: Courier, monospace;
    font-size: 8px;
    background: #f4f4f4;
    padding: 1px 3px;
}
.table-wrap {
    margin: 8px 0;
    overflow: hidden;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0;
}
th {
    background-color: #3498db;
    color: white;
    font-weight: bold;
    padding: 4px 5px;
    text-align: center;
    border: 1px solid #2980b9;
    word-wrap: break-word;
}
td {
    padding: 3px 5px;
    border: 1px solid #ddd;
    text-align: center;
    word-wrap: break-word;
}
/* Index column cells (converted from th to td.index-cell in post-processing) */
td.index-cell {
    background-color: #eaf2f8;
    color: #2c3e50;
    font-weight: bold;
    text-align: center;
    padding: 3px 5px;
    border: 1px solid #ddd;
    word-wrap: break-word;
    overflow: hidden;
}

/* Standard tables (7 cols or fewer) */
.std-table th, .std-table td {
    font-size: 8px;
}

/* Mid tables (8-10 cols) */
.mid-table th, .mid-table td {
    font-size: 7px;
    padding: 3px 4px;
}

/* Wide tables (11+ cols like describe(include='all')) */
.wide-table th, .wide-table td {
    font-size: 6px;
    padding: 2px 3px;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 6px auto;
}
hr {
    border: none;
    border-top: 1px solid #bdc3c7;
    margin: 12px 0;
}
</style>
"""

html = html.replace('</head>', report_css + '</head>')

# Clean up nbconvert-specific class names
html = re.sub(r'<div class="jp-Cell[^"]*">', '<div>', html)
html = re.sub(r'<div class="jp-RenderedHTMLCommon[^"]*">', '<div>', html)
html = re.sub(r'<div class="jp-InputArea[^"]*">', '<div>', html)
html = re.sub(r'<div class="jp-OutputArea[^"]*">', '<div>', html)

# Step 6: Convert to PDF
print("Generating PDF...")
from xhtml2pdf import pisa

with open(PDF_PATH, "wb") as pdf_file:
    status = pisa.CreatePDF(html, dest=pdf_file)

if status.err:
    print(f"PDF had {status.err} errors")
else:
    size_kb = os.path.getsize(PDF_PATH) // 1024
    print(f"PDF saved: {PDF_PATH} ({size_kb} KB)")

# Clean up intermediate HTML
if os.path.exists(HTML_PATH):
    os.remove(HTML_PATH)
    print("Cleaned up intermediate HTML.")

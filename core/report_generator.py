"""
Report Generator Module - Creates HTML and PDF reports.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any


def generate_html_report(df: pd.DataFrame, insights_text: str, charts_html: str = "",
                         title: str = "Data Analytics Report") -> str:
    """Generate an HTML report."""
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f5f7fa; color: #333; padding: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); padding: 40px; }}
        h1 {{ color: #667eea; margin-bottom: 10px; }}
        h2 {{ color: #764ba2; margin: 30px 0 15px; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        h3 {{ color: #555; margin: 20px 0 10px; }}
        .meta {{ color: #888; margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
        th {{ background: linear-gradient(135deg, #667eea, #764ba2); color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .insight-box {{ background: #f0f4ff; border-left: 4px solid #667eea; padding: 15px; margin: 15px 0; border-radius: 0 8px 8px 0; }}
        .chart-container {{ margin: 20px 0; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }}
        ul {{ padding-left: 25px; margin: 10px 0; }}
        li {{ margin: 5px 0; }}
        strong {{ color: #444; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 {title}</h1>
    <p class="meta">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Dataset Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Rows</td><td>{len(df):,}</td></tr>
        <tr><td>Total Columns</td><td>{len(df.columns)}</td></tr>
        <tr><td>Memory Usage</td><td>{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB</td></tr>
    </table>
    
    <h2>Column Overview</h2>
    {df.dtypes.to_frame('Data Type').to_html()}
    
    <h2>Insights</h2>
    <div class="insight-box">
        {insights_text.replace(chr(10), '<br>').replace('###', '<h3>').replace('**', '<strong>')}
    </div>
    
    {charts_html}
    
</div>
</body>
</html>"""
    
    return html


def save_html_report(html_content: str, filepath: str) -> bool:
    """Save HTML report to file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return True
    except Exception:
        return False


def generate_pdf_from_html(html_content: str, filepath: str) -> bool:
    """
    Generate PDF from HTML code.
    Tries 'weasyprint' first (high quality), falls back to 'xhtml2pdf' (compatibility).
    """
    # 1. Try WeasyPrint (Best Quality, requires GTK3)
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(filepath)
        return True
    except (ImportError, OSError):
        pass # Fall through to next method

    # 2. Try xhtml2pdf (Good Compatibility, pure python-ish)
    try:
        from xhtml2pdf import pisa
        with open(filepath, "wb") as result_file:
            # pisa requires bytes or string, handle encoding carefully
            pisa_status = pisa.CreatePDF(
                html_content,    # the HTML to convert
                dest=result_file # file handle to recieve result
            )
        return not pisa_status.err
    except ImportError:
        return False
    except Exception as e:
        print(f"PDF generation failed: {e}")
        return False

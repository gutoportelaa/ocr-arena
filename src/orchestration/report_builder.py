import pandas as pd
from pathlib import Path
from typing import Dict, Any
from loguru import logger

class ReportBuilder:
    def __init__(self, reports_dir: str):
        self.reports_dir = Path(reports_dir)

    def build_markdown_report(self, summaries: Dict[str, pd.DataFrame], routing_rules: str) -> str:
        logger.info("Building final markdown report")
        
        report = "# OCR Benchmark Final Report\n\n"
        
        report += "## 1. Engine Performance Summary\n\n"
        report += summaries["by_engine"].to_markdown() + "\n\n"
        
        report += "## 2. Performance by Document\n\n"
        report += summaries["by_pdf"].head(20).to_markdown() + "\n\n"
        
        report += "## 3. Recommended Routing Pipeline\n\n"
        report += routing_rules + "\n\n"
        
        report += "## 4. Methodology\n\n"
        report += "- 20 PDFs selected by complexity score.\n"
        report += "- Benchmarked on local GPU (RTX 4070).\n"
        report += "- Metrics include success rate, execution time, and semantic similarity.\n"
        
        return report

    def save_report(self, content: str, filename: str = "final_report.md"):
        path = self.reports_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Report saved to {path}")

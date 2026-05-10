from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from src.engines.base_engine import BaseOCREngine, OCRResult

class MarkItDownEngine(BaseOCREngine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.enabled = config.get("enabled", False)

    def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        if not self.enabled:
            return [OCRResult(self.name, pdf_path.stem, None, "", "", {}, False, "unsupported_or_partial")]
        
        logger.info(f"MarkItDown processing PDF: {pdf_path.name}")
        return []

    def process_page(self, image_path: Path, page_number: int, pdf_name: str) -> OCRResult:
        return OCRResult(self.name, pdf_name, page_number, "", "", {}, False, "unsupported_or_partial")

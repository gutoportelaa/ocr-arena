import pytesseract
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from src.engines.base_engine import BaseOCREngine, OCRResult
from src.utils.time_utils import Timer

class TesseractEngine(BaseOCREngine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.tesseract_path = config.get("path")
        self.lang = config.get("language", "por+eng")
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

    def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        return []

    def process_page(self, image_path: Path, page_number: int, pdf_name: str) -> OCRResult:
        logger.info(f"Tesseract processing page {page_number} of {pdf_name}")
        
        try:
            with Timer() as t:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image, lang=self.lang)
            
            return OCRResult(
                engine_name=self.name,
                pdf_name=pdf_name,
                page_number=page_number,
                text_plain=text,
                text_markdown=text, # Tesseract output is plain text
                metadata={"lang": self.lang},
                success=True,
                execution_time=t.duration
            )
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return OCRResult(
                engine_name=self.name,
                pdf_name=pdf_name,
                page_number=page_number,
                text_plain="",
                text_markdown="",
                metadata={},
                success=False,
                error_message=str(e)
            )

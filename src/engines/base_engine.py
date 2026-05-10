from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class OCRResult:
    engine_name: str
    pdf_name: str
    page_number: Optional[int]
    text_plain: str
    text_markdown: str
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0

class BaseOCREngine(ABC):
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    @abstractmethod
    def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        """Process an entire PDF file."""
        pass

    @abstractmethod
    def process_page(self, image_path: Path, page_number: int, pdf_name: str) -> OCRResult:
        """Process a single page image."""
        pass

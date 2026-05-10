import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import fitz  # PyMuPDF
from src.utils.hash_utils import get_file_sha256
from src.utils.io_utils import list_files

class PDFInventory:
    def __init__(self, raw_dir: str):
        self.raw_dir = Path(raw_dir)

    def scan(self) -> pd.DataFrame:
        logger.info(f"Scanning directory: {self.raw_dir}")
        pdf_files = list_files(self.raw_dir, extensions=[".pdf"])
        inventory_data = []

        for pdf_path in pdf_files:
            try:
                info = self._get_pdf_info(pdf_path)
                inventory_data.append(info)
            except Exception as e:
                logger.error(f"Error scanning {pdf_path}: {e}")

        df = pd.DataFrame(inventory_data)
        return df

    def _get_pdf_info(self, path: Path) -> Dict[str, Any]:
        doc = fitz.open(path)
        num_pages = len(doc)
        
        # Estimate features (simplified for this implementation)
        total_images = 0
        total_native_text_len = 0
        page_sizes = []
        
        for page in doc:
            total_images += len(page.get_images())
            total_native_text_len += len(page.get_text().strip())
            rect = page.rect
            page_sizes.append((rect.width, rect.height))

        avg_width = sum(p[0] for p in page_sizes) / num_pages if num_pages > 0 else 0
        avg_height = sum(p[1] for p in page_sizes) / num_pages if num_pages > 0 else 0
        
        doc.close()

        return {
            "filename": path.name,
            "path": str(path),
            "sha256": get_file_sha256(path),
            "file_size_kb": path.stat().st_size / 1024,
            "page_count": num_pages,
            "avg_page_width": avg_width,
            "avg_page_height": avg_height,
            "native_text_length": total_native_text_len,
            "image_count": total_images,
            "native_text_density": total_native_text_len / num_pages if num_pages > 0 else 0,
            "image_density": total_images / num_pages if num_pages > 0 else 0,
            # Placeholder for more complex features
            "scan_ratio": 0.5 if total_native_text_len < 100 else 0.1,
            "table_density": 0.1,
            "rotation_irregularity": 0.0,
            "multi_column_likelihood": 0.2,
            "language_uncertainty": 0.1
        }

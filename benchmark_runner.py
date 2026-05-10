import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from src.engines.base_engine import BaseOCREngine, OCRResult
from src.ingestion.pdf_render import PDFRenderer
from src.utils.io_utils import ensure_dir, write_text
import json

class BenchmarkRunner:
    def __init__(self, engines: List[BaseOCREngine], config: Dict[str, Any]):
        self.engines = engines
        self.config = config
        self.renderer = PDFRenderer(
            output_dir=config["data_base_dir"] + "/rendered_pages",
            dpi=config.get("dpi_render", 300)
        )

    def run(self, selected_pdfs: pd.DataFrame) -> List[OCRResult]:
        all_results = []
        
        for _, pdf_row in selected_pdfs.iterrows():
            pdf_path = Path(pdf_row["path"])
            pdf_name = pdf_path.stem
            logger.info(f"Processing PDF: {pdf_name}")
            
            # Render pages once for all engines that need them
            page_images = self.renderer.render_pdf(str(pdf_path))
            
            for engine in self.engines:
                logger.info(f"Running engine: {engine.name} on {pdf_name}")
                
                # Try PDF-level processing first
                engine_results = engine.process_pdf(pdf_path)
                
                # If engine didn't return results, try page-level
                if not engine_results:
                    for i, img_path in enumerate(page_images):
                        page_num = i + 1
                        result = engine.process_page(img_path, page_num, pdf_name)
                        self._save_result(result)
                        all_results.append(result)
                else:
                    for result in engine_results:
                        self._save_result(result)
                        all_results.append(result)
                        
        return all_results

    def _save_result(self, result: OCRResult):
        output_dir = Path(self.config["output_base_dir"]) / "extracted" / result.engine_name / result.pdf_name
        ensure_dir(output_dir)
        
        filename_base = f"page_{result.page_number:03d}" if result.page_number else "full"
        
        # Save text outputs
        write_text(output_dir / f"{filename_base}.txt", result.text_plain)
        write_text(output_dir / f"{filename_base}.md", result.text_markdown)
        
        # Save metadata
        meta_path = output_dir / f"{filename_base}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "engine": result.engine_name,
                "pdf": result.pdf_name,
                "page": result.page_number,
                "success": result.success,
                "error": result.error_message,
                "execution_time": result.execution_time,
                "metadata": result.metadata
            }, f, indent=2)

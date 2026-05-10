from pathlib import Path
from pdf2image import convert_from_path
from loguru import logger
from src.utils.io_utils import ensure_dir

class PDFRenderer:
    def __init__(self, output_dir: str, dpi: int = 300):
        self.output_dir = Path(output_dir)
        self.dpi = dpi

    def render_pdf(self, pdf_path: str) -> list[Path]:
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        target_dir = self.output_dir / pdf_name
        ensure_dir(target_dir)
        
        logger.info(f"Rendering PDF: {pdf_path.name} to {target_dir}")
        
        images = convert_from_path(pdf_path, dpi=self.dpi)
        image_paths = []
        
        for i, image in enumerate(images):
            page_num = i + 1
            image_path = target_dir / f"page_{page_num:03d}.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
            
        return image_paths

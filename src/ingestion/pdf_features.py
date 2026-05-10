import fitz
from typing import Dict, Any

def extract_page_features(pdf_path: str, page_num: int) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num - 1)
    
    text = page.get_text()
    images = page.get_images()
    tables = page.find_tables()
    
    features = {
        "native_text_len": len(text.strip()),
        "image_count": len(images),
        "table_count": len(tables.tables) if tables else 0,
        "is_scanned": len(text.strip()) < 50 and len(images) > 0,
        "width": page.rect.width,
        "height": page.rect.height
    }
    
    doc.close()
    return features

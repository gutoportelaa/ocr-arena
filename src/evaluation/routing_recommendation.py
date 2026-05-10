from typing import Dict, Any

def recommend_engine(pdf_features: Dict[str, Any], engine_performance: Dict[str, float]) -> str:
    """
    Recommend an OCR engine based on document features and historical performance.
    
    Logic:
    - If scanned and high image density -> Prefer Vision-based models (Ollama/olmocr)
    - If native text and complex layout -> Prefer structured parsers (Docling/Marker)
    - If simple text -> Prefer fast CPU-based (Tesseract)
    """
    is_scanned = pdf_features.get("is_scanned", False)
    image_density = pdf_features.get("image_density", 0)
    native_text_density = pdf_features.get("native_text_density", 0)
    
    if is_scanned or image_density > 0.5:
        return "ollama_olmocr2"
    elif native_text_density > 1000:
        return "marker"
    else:
        return "tesseract"

def generate_routing_rules(benchmark_results: Any) -> str:
    """Generate human-readable routing rules based on benchmark data."""
    # This would analyze the benchmark_results and produce a logic tree
    return """
    # Proposed Routing Logic
    1. IF document is highly complex (complexity_score > 0.8) -> ROUTE TO Ollama (olmocr2)
    2. IF document has high native text density (> 0.9) -> ROUTE TO Marker/Docling
    3. IF document is low complexity AND mostly text -> ROUTE TO Tesseract
    4. DEFAULT -> ROUTE TO Ollama (MiniCPM-V)
    """

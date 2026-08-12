import torch
import tempfile
import os
import fitz # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from src.engines.base_engine import BaseOCREngine, OCRResult
from src.utils.time_utils import Timer
import re

DET_RE = re.compile(r'<\|det\|>([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>(.*)', re.DOTALL)

def remove_det(raw: str) -> str:
    """
    Strip <|det|>type [bbox]<|/det|> markers, group lines belonging to the
    same block with \n, and separate different blocks with \n\n.
    """
    blocks = []
    cur = None
    for line in raw.splitlines():
        line = line.rstrip()
        if not line:
            continue
        m = DET_RE.match(line)
        if m:
            category, content = m.group(1).strip(), m.group(2).strip()
            if category == 'image':
                continue
            if cur is not None:
                blocks.append(cur)
            cur = [content] if content else []
            continue
        if cur is None:
            cur = []
        cur.append(line)
    if cur is not None:
        blocks.append(cur)
    text = '\n\n'.join('\n'.join(b) for b in blocks).strip()
    return text

class UnlimitedOCREngine(BaseOCREngine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.model_name = config.get("model_name", "baidu/Unlimited-OCR")
        self.torch_dtype_str = config.get("torch_dtype", "bfloat16")
        
        self.torch_dtype = torch.bfloat16 if self.torch_dtype_str == "bfloat16" else torch.float16
        
        logger.info(f"Loading {self.model_name} with dtype {self.torch_dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_safetensors=True,
            torch_dtype=self.torch_dtype,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.eval().to(self.device)
        logger.info(f"Successfully loaded {self.model_name} on {self.device}")

    def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        logger.info(f"UnlimitedOCR processing entire PDF: {pdf_path.name}")
        results = []
        try:
            with Timer() as t:
                # Convert PDF to images
                doc = fitz.open(pdf_path)
                tmp_dir = tempfile.mkdtemp(prefix='pdf_ocr_')
                mat = fitz.Matrix(300 / 72, 300 / 72)
                image_paths = []
                for i, page in enumerate(doc):
                    out = os.path.join(tmp_dir, f'page_{i+1:04d}.png')
                    page.get_pixmap(matrix=mat).save(out)
                    image_paths.append(out)
                doc.close()
                
                # Perform inference
                # Note: infer_multi returns a list or a concatenated string depending on implementation.
                # Assuming it generates markdown content per page or single output. 
                # According to docs, save_results=False might return raw text. Let's do it per page for granular OCRResult.
                for idx, image_path in enumerate(image_paths):
                    page_num = idx + 1
                    res = self.process_page(Path(image_path), page_num, pdf_path.name)
                    results.append(res)
                    
        except Exception as e:
            logger.error(f"UnlimitedOCR error on {pdf_path.name}: {e}")
            
        return results

    def process_page(self, image_path: Path, page_number: int, pdf_name: str) -> OCRResult:
        logger.info(f"UnlimitedOCR processing page {page_number} of {pdf_name}")
        
        try:
            with Timer() as t:
                # Use model.infer for single page
                # We need to capture the output, the README uses save_results=True, output_path='...'
                # We will save to a temporary directory and read the result, or intercept the return value.
                # Let's save to a temp dir and read.
                with tempfile.TemporaryDirectory() as tmp_out:
                    self.model.infer(
                        self.tokenizer,
                        prompt='<image>document parsing.',
                        image_file=str(image_path),
                        output_path=tmp_out,
                        base_size=1024, image_size=640, crop_mode=True,
                        max_length=32768,
                        no_repeat_ngram_size=35, ngram_window=128,
                        save_results=True,
                    )
                    
                    # Read the generated markdown
                    # model.infer typically saves a .md file or similar with the base name of the image
                    base_name = image_path.stem
                    md_path = Path(tmp_out) / f"{base_name}.md"
                    
                    if md_path.exists():
                        raw_text = md_path.read_text(encoding='utf-8')
                    else:
                        # try txt just in case
                        txt_path = Path(tmp_out) / f"{base_name}.txt"
                        if txt_path.exists():
                            raw_text = txt_path.read_text(encoding='utf-8')
                        else:
                            # fallback: list all files
                            files = list(Path(tmp_out).glob("*"))
                            if files:
                                raw_text = files[0].read_text(encoding='utf-8')
                            else:
                                raw_text = ""
                
                # Apply the specific post-processing for OmniDocBench to clean det tags
                clean_text = remove_det(raw_text)

            return OCRResult(
                engine_name=self.name,
                pdf_name=pdf_name,
                page_number=page_number,
                text_plain=clean_text,
                text_markdown=raw_text,
                metadata={"model": self.model_name},
                success=True,
                execution_time=t.duration
            )
        except Exception as e:
            logger.error(f"UnlimitedOCR error: {e}")
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

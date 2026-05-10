import ollama
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from src.engines.base_engine import BaseOCREngine, OCRResult
from src.utils.time_utils import Timer

class OllamaEngine(BaseOCREngine):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.model_name = config.get("name")
        self.prompt_template = config.get("prompt_template")
        self.json_schema = config.get("json_schema")

    def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        # OllamaEngine usually processes pages as images
        # This will be handled by the orchestrator
        return []

    def process_page(self, image_path: Path, page_number: int, pdf_name: str) -> OCRResult:
        logger.info(f"Ollama {self.model_name} processing page {page_number} of {pdf_name}")
        
        try:
            with Timer() as t:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=self.prompt_template,
                    images=[str(image_path)],
                    options={"temperature": 0},
                    format="json" if self.json_schema else None
                )
            
            output_text = response.get("response", "")
            
            # Parse structured output if schema was provided
            text_plain = output_text
            text_markdown = output_text
            metadata = {
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count")
            }

            if self.json_schema:
                try:
                    data = json.loads(output_text)
                    text_plain = data.get("text_plain", output_text)
                    text_markdown = data.get("text_markdown", output_text)
                    metadata.update(data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response from Ollama for {pdf_name} page {page_number}")

            return OCRResult(
                engine_name=self.name,
                pdf_name=pdf_name,
                page_number=page_number,
                text_plain=text_plain,
                text_markdown=text_markdown,
                metadata=metadata,
                success=True,
                execution_time=t.duration
            )

        except Exception as e:
            logger.error(f"Ollama error: {e}")
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

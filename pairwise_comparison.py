import json
from typing import Dict, Any
from loguru import logger
import ollama

class PairwiseEvaluator:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def compare(self, text1: str, text2: str, prompt_template: str) -> Dict[str, Any]:
        """Use a local LLM to compare two OCR outputs."""
        prompt = f"Output 1:\n{text1}\n\nOutput 2:\n{text2}\n\n{prompt_template}"
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format="json"
            )
            return json.loads(response.get("response", "{}"))
        except Exception as e:
            logger.error(f"Pairwise comparison error: {e}")
            return {"error": str(e)}

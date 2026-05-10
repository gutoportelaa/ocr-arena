import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.models = self._load_yaml("models.yaml")
        self.benchmark = self._load_yaml("benchmark.yaml")
        self.prompts = self._load_yaml("prompts.yaml")
        
        # Override with environment variables if present
        self._apply_env_overrides()

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _apply_env_overrides(self):
        # Example override logic
        data_dir = os.getenv("DATA_DIR", "./data")
        self.benchmark["pdf_selection"]["raw_pdfs_dir"] = self.benchmark["pdf_selection"]["raw_pdfs_dir"].replace("${DATA_DIR}", data_dir)
        self.benchmark["pdf_selection"]["selected_pdfs_dir"] = self.benchmark["pdf_selection"]["selected_pdfs_dir"].replace("${DATA_DIR}", data_dir)
        
        # Add more overrides as needed for other variables like TESSERACT_PATH, etc.
        tesseract_path = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")
        if "external_engines" in self.models and "tesseract" in self.models["external_engines"]:
            self.models["external_engines"]["tesseract"]["path"] = self.models["external_engines"]["tesseract"]["path"].replace("${TESSERACT_PATH}", tesseract_path)

    def get_model_config(self, model_type: str, model_name: str) -> Dict[str, Any]:
        if model_type == "ollama":
            return self.models.get("ollama_models", {}).get(model_name, {})
        elif model_type == "external":
            return self.models.get("external_engines", {}).get(model_name, {})
        return {}

    def get_benchmark_setting(self, key: str, default: Any = None) -> Any:
        return self.benchmark.get("benchmark_settings", {}).get(key, default)

    def get_prompt(self, category: str, name: str) -> str:
        return self.prompts.get(category, {}).get(name, "")

config = Config()

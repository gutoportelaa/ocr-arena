import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json
from loguru import logger

class ResultAggregator:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def aggregate(self) -> pd.DataFrame:
        logger.info("Aggregating results from disk")
        extracted_dir = self.output_dir / "extracted"
        
        all_data = []
        
        for engine_dir in extracted_dir.iterdir():
            if not engine_dir.is_dir(): continue
            
            for pdf_dir in engine_dir.iterdir():
                if not pdf_dir.is_dir(): continue
                
                for meta_file in pdf_dir.glob("*_meta.json"):
                    with open(meta_file, "r") as f:
                        data = json.load(f)
                        all_data.append({
                            "engine": data["engine"],
                            "pdf": data["pdf"],
                            "page": data["page"],
                            "success": data["success"],
                            "execution_time": data["execution_time"],
                            "text_len": len(data.get("metadata", {}).get("text_plain", "")) # or read from .txt
                        })
        
        df = pd.DataFrame(all_data)
        return df

    def create_summaries(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        summaries = {}
        
        # Summary by engine
        summaries["by_engine"] = df.groupby("engine").agg({
            "success": "mean",
            "execution_time": ["mean", "sum"],
            "pdf": "nunique"
        })
        
        # Summary by PDF
        summaries["by_pdf"] = df.groupby(["pdf", "engine"]).agg({
            "success": "mean",
            "execution_time": "sum"
        })
        
        return summaries

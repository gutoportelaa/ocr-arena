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

        for engine_dir in sorted(extracted_dir.iterdir()):
            if not engine_dir.is_dir():
                continue

            for pdf_dir in sorted(engine_dir.iterdir()):
                if not pdf_dir.is_dir():
                    continue

                for meta_file in sorted(pdf_dir.glob("*_meta.json")):
                    with open(meta_file, "r") as f:
                        data = json.load(f)

                    # Read actual text length from sibling .txt file
                    txt_file = meta_file.with_name(meta_file.stem.replace("_meta", "") + ".txt")
                    chars_extracted = len(txt_file.read_text(encoding="utf-8")) if txt_file.exists() else 0

                    all_data.append(
                        {
                            "engine": data["engine"],
                            "pdf": data["pdf"],
                            "page": data["page"],
                            "success": data["success"],
                            "execution_time": data.get("execution_time", 0.0),
                            "chars_extracted": chars_extracted,
                            "error": data.get("error") or "",
                        }
                    )

        df = pd.DataFrame(all_data)
        return df

    def create_summaries(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        summaries = {}

        if df.empty:
            summaries["by_engine"] = pd.DataFrame()
            summaries["by_pdf"] = pd.DataFrame()
            summaries["timing_comparison"] = pd.DataFrame()
            return summaries

        # Per-engine aggregate
        by_engine = (
            df.groupby("engine")
            .agg(
                success_rate=("success", "mean"),
                avg_time_per_page=("execution_time", "mean"),
                total_time=("execution_time", "sum"),
                avg_chars_extracted=("chars_extracted", "mean"),
                total_chars_extracted=("chars_extracted", "sum"),
                pages_processed=("page", "count"),
                pdfs_processed=("pdf", "nunique"),
            )
            .round(3)
            .sort_values("avg_time_per_page")
        )
        by_engine["success_rate_%"] = (by_engine["success_rate"] * 100).round(1)
        summaries["by_engine"] = by_engine

        # Per-PDF, per-engine aggregate
        summaries["by_pdf"] = (
            df.groupby(["pdf", "engine"])
            .agg(
                success_rate=("success", "mean"),
                total_time=("execution_time", "sum"),
                total_chars=("chars_extracted", "sum"),
            )
            .round(3)
        )

        # Timing comparison table (engines as columns, PDFs as rows)
        if "pdf" in df.columns and "engine" in df.columns:
            timing_pivot = df.groupby(["pdf", "engine"])["execution_time"].sum().unstack(fill_value=0)
            timing_pivot.columns = [f"{c}_time_s" for c in timing_pivot.columns]
            summaries["timing_comparison"] = timing_pivot.round(2)

            chars_pivot = df.groupby(["pdf", "engine"])["chars_extracted"].sum().unstack(fill_value=0)
            chars_pivot.columns = [f"{c}_chars" for c in chars_pivot.columns]
            summaries["chars_comparison"] = chars_pivot

        # Docling CPU vs GPU speedup (if both engines present)
        docling_engines = [e for e in df["engine"].unique() if "docling" in e.lower()]
        if "docling" in docling_engines and "docling_gpu" in docling_engines:
            cpu_time = df[df["engine"] == "docling"]["execution_time"].sum()
            gpu_time = df[df["engine"] == "docling_gpu"]["execution_time"].sum()
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("nan")
            summaries["docling_speedup"] = {
                "docling_cpu_total_s": round(cpu_time, 2),
                "docling_gpu_total_s": round(gpu_time, 2),
                "gpu_speedup_factor": round(speedup, 2),
            }

        return summaries

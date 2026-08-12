import argparse
import pandas as pd
from pathlib import Path
from loguru import logger

from src.config import config
from src.utils.logging_utils import setup_logging, log_system_info
from src.ingestion.pdf_inventory import PDFInventory
from src.ingestion.pdf_sampling import PDFSampler
from src.ingestion.pdf_render import PDFRenderer
from src.engines.ollama_engine import OllamaEngine
from src.engines.tesseract_engine import TesseractEngine
from src.engines.docling_engine import DoclingEngine
from src.engines.marker_engine import MarkerEngine
from src.engines.markitdown_engine import MarkItDownEngine
from src.engines.unlimited_ocr_engine import UnlimitedOCREngine
from src.orchestration.benchmark_runner import BenchmarkRunner
from src.orchestration.result_aggregator import ResultAggregator
from src.orchestration.report_builder import ReportBuilder
from src.evaluation.routing_recommendation import generate_routing_rules

def main():
    parser = argparse.ArgumentParser(description="OCR Benchmark Tool")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml", help="Path to benchmark configuration file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inventory command
    inventory_parser = subparsers.add_parser("inventory", help="Create PDF inventory")
    inventory_parser.add_argument("--raw_dir", type=str, default=config.benchmark["pdf_selection"]["raw_pdfs_dir"], help="Directory with raw PDFs")
    inventory_parser.add_argument("--output", type=str, default="reports/pdf_inventory.csv", help="Output CSV file for inventory")

    # Select command
    select_parser = subparsers.add_parser("select", help="Select 20 PDFs based on complexity")
    select_parser.add_argument("--inventory", type=str, default="reports/pdf_inventory.csv", help="Input CSV inventory file")
    select_parser.add_argument("--output", type=str, default="reports/selected_20.csv", help="Output CSV file for selected PDFs")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run OCR benchmark on selected PDFs")
    benchmark_parser.add_argument("--selected_pdfs", type=str, default="reports/selected_20.csv", help="Input CSV file with selected PDFs")

    # Aggregate command
    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate benchmark results")
    aggregate_parser.add_argument("--output", type=str, default="reports/benchmark_results.csv", help="Output CSV file for aggregated results")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate final report")
    report_parser.add_argument("--aggregated_results", type=str, default="reports/benchmark_results.csv", help="Input CSV file with aggregated results")
    report_parser.add_argument("--output", type=str, default="reports/final_report.md", help="Output Markdown file for the report")

    # All command
    all_parser = subparsers.add_parser("all", help="Run full benchmark pipeline")
    all_parser.add_argument("--raw_dir", type=str, default=config.benchmark["pdf_selection"]["raw_pdfs_dir"], help="Directory with raw PDFs")
    all_parser.add_argument("--inventory_output", type=str, default="reports/pdf_inventory.csv", help="Output CSV file for inventory")
    all_parser.add_argument("--selected_output", type=str, default="reports/selected_20.csv", help="Output CSV file for selected PDFs")
    all_parser.add_argument("--aggregated_output", type=str, default="reports/benchmark_results.csv", help="Output CSV file for aggregated results")
    report_parser.add_argument("--report_output", type=str, default="reports/final_report.md", help="Output Markdown file for the report")

    args = parser.parse_args()

    setup_logging(log_dir=config.benchmark["benchmark_settings"]["log_base_dir"], level=config.get_benchmark_setting("log_level", "INFO"))
    log_system_info()

    if args.command == "inventory":
        inventory = PDFInventory(args.raw_dir)
        df_inventory = inventory.scan()
        df_inventory.to_csv(args.output, index=False)
        logger.info(f"PDF inventory saved to {args.output}")

    elif args.command == "select":
        df_inventory = pd.read_csv(args.inventory)
        sampler = PDFSampler(config.benchmark["pdf_selection"]["complexity_weights"])
        df_complexity = sampler.calculate_complexity(df_inventory)
        df_selected = sampler.select_top_n(
            df_complexity,
            n_high=config.benchmark["pdf_selection"]["num_high_complexity"],
            n_med=config.benchmark["pdf_selection"]["num_medium_complexity"],
            n_low=config.benchmark["pdf_selection"]["num_low_complexity"],
            n_outliers=config.benchmark["pdf_selection"]["num_outliers"]
        )
        df_selected.to_csv(args.output, index=False)
        logger.info(f"Selected PDFs saved to {args.output}")

    elif args.command == "benchmark":
        selected_pdfs = pd.read_csv(args.selected_pdfs)
        
        engines = []
        for model_name, model_config in config.models.get("ollama_models", {}).items():
            if model_config.get("enabled", True):
                engines.append(OllamaEngine(model_name, model_config))
        
        tesseract_config = config.models.get("external_engines", {}).get("tesseract", {})
        if tesseract_config.get("enabled", False):
            engines.append(TesseractEngine("tesseract", tesseract_config))

        docling_config = config.models.get("external_engines", {}).get("docling", {})
        if docling_config.get("enabled", False):
            engines.append(DoclingEngine("docling", docling_config))

        docling_gpu_config = config.models.get("external_engines", {}).get("docling_gpu", {})
        if docling_gpu_config.get("enabled", False):
            engines.append(DoclingEngine("docling_gpu", docling_gpu_config))

        marker_config = config.models.get("external_engines", {}).get("marker", {})
        if marker_config.get("enabled", False):
            engines.append(MarkerEngine("marker", marker_config))

        markitdown_config = config.models.get("external_engines", {}).get("markitdown", {})
        if markitdown_config.get("enabled", False):
            engines.append(MarkItDownEngine("markitdown", markitdown_config))

        unlimited_ocr_config = config.models.get("external_engines", {}).get("unlimited_ocr", {})
        if unlimited_ocr_config.get("enabled", False):
            engines.append(UnlimitedOCREngine("unlimited_ocr", unlimited_ocr_config))

        runner = BenchmarkRunner(engines, config.benchmark["benchmark_settings"])
        all_results = runner.run(selected_pdfs)
        # Results are saved to disk by the runner, no need to return here
        logger.info("Benchmark run completed.")

    elif args.command == "aggregate":
        aggregator = ResultAggregator(config.benchmark["benchmark_settings"]["output_base_dir"])
        df_aggregated = aggregator.aggregate()
        df_aggregated.to_csv(args.output, index=False)
        logger.info(f"Aggregated results saved to {args.output}")

    elif args.command == "report":
        df_aggregated = pd.read_csv(args.aggregated_results)
        aggregator = ResultAggregator(config.benchmark["benchmark_settings"]["output_base_dir"])
        summaries = aggregator.create_summaries(df_aggregated)
        routing_rules = generate_routing_rules(df_aggregated) # This needs actual implementation
        
        report_builder = ReportBuilder(config.benchmark["benchmark_settings"]["reports_base_dir"])
        markdown_report = report_builder.build_markdown_report(summaries, routing_rules)
        report_builder.save_report(markdown_report, filename=Path(args.output).name)
        logger.info(f"Final report generated at {args.output}")

    elif args.command == "all":
        logger.info("Running full benchmark pipeline...")
        # 1. Inventory
        inventory = PDFInventory(args.raw_dir)
        df_inventory = inventory.scan()
        df_inventory.to_csv(args.inventory_output, index=False)
        logger.info(f"PDF inventory saved to {args.inventory_output}")

        # 2. Select PDFs
        sampler = PDFSampler(config.benchmark["pdf_selection"]["complexity_weights"])
        df_complexity = sampler.calculate_complexity(df_inventory)
        df_selected = sampler.select_top_n(
            df_complexity,
            n_high=config.benchmark["pdf_selection"]["num_high_complexity"],
            n_med=config.benchmark["pdf_selection"]["num_medium_complexity"],
            n_low=config.benchmark["pdf_selection"]["num_low_complexity"],
            n_outliers=config.benchmark["pdf_selection"]["num_outliers"]
        )
        df_selected.to_csv(args.selected_output, index=False)
        logger.info(f"Selected PDFs saved to {args.selected_output}")

        # 3. Benchmark
        engines = []
        for model_name, model_config in config.models.get("ollama_models", {}).items():
            if model_config.get("enabled", True):
                engines.append(OllamaEngine(model_name, model_config))
        
        tesseract_config = config.models.get("external_engines", {}).get("tesseract", {})
        if tesseract_config.get("enabled", False):
            engines.append(TesseractEngine("tesseract", tesseract_config))

        docling_config = config.models.get("external_engines", {}).get("docling", {})
        if docling_config.get("enabled", False):
            engines.append(DoclingEngine("docling", docling_config))

        docling_gpu_config = config.models.get("external_engines", {}).get("docling_gpu", {})
        if docling_gpu_config.get("enabled", False):
            engines.append(DoclingEngine("docling_gpu", docling_gpu_config))

        marker_config = config.models.get("external_engines", {}).get("marker", {})
        if marker_config.get("enabled", False):
            engines.append(MarkerEngine("marker", marker_config))

        markitdown_config = config.models.get("external_engines", {}).get("markitdown", {})
        if markitdown_config.get("enabled", False):
            engines.append(MarkItDownEngine("markitdown", markitdown_config))

        unlimited_ocr_config = config.models.get("external_engines", {}).get("unlimited_ocr", {})
        if unlimited_ocr_config.get("enabled", False):
            engines.append(UnlimitedOCREngine("unlimited_ocr", unlimited_ocr_config))

        runner = BenchmarkRunner(engines, config.benchmark["benchmark_settings"])
        runner.run(df_selected)
        logger.info("Benchmark run completed.")

        # 4. Aggregate
        aggregator = ResultAggregator(config.benchmark["benchmark_settings"]["output_base_dir"])
        df_aggregated = aggregator.aggregate()
        df_aggregated.to_csv(args.aggregated_output, index=False)
        logger.info(f"Aggregated results saved to {args.aggregated_output}")

        # 5. Report
        summaries = aggregator.create_summaries(df_aggregated)
        routing_rules = generate_routing_rules(df_aggregated)
        
        report_builder = ReportBuilder(config.benchmark["benchmark_settings"]["reports_base_dir"])
        markdown_report = report_builder.build_markdown_report(summaries, routing_rules)
        report_builder.save_report(markdown_report, filename=Path(args.report_output).name)
        logger.info(f"Final report generated at {args.report_output}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()

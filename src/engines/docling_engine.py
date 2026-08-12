from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

from src.engines.base_engine import BaseOCREngine, OCRResult
from src.utils.time_utils import Timer


class DoclingEngine(BaseOCREngine):
    """Docling engine with optional GPU acceleration via CUDA."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.enabled = config.get("enabled", False)
        self.use_gpu = config.get("use_gpu", False)
        self._converter = None

    def _build_converter(self):
        """Lazy-build the DocumentConverter so import errors surface at runtime."""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                AcceleratorOptions,
                AcceleratorDevice,
            )
            from docling.datamodel.base_models import InputFormat
        except ImportError as e:
            raise ImportError(
                "docling is not installed. Run: pip install docling"
            ) from e

        device = AcceleratorDevice.CUDA if self.use_gpu else AcceleratorDevice.CPU
        accelerator_opts = AcceleratorOptions(
            num_threads=4,
            device=device,
        )
        pipeline_opts = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            accelerator_options=accelerator_opts,
        )
        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )
        logger.info(
            f"[{self.name}] Docling converter initialised "
            f"(device={'CUDA' if self.use_gpu else 'CPU'})"
        )

    def process_pdf(self, pdf_path: Path) -> List[OCRResult]:
        if not self.enabled:
            return [
                OCRResult(
                    engine_name=self.name,
                    pdf_name=pdf_path.stem,
                    page_number=None,
                    text_plain="",
                    text_markdown="",
                    metadata={},
                    success=False,
                    error_message="Engine disabled",
                )
            ]

        if self._converter is None:
            try:
                self._build_converter()
            except ImportError as e:
                return [
                    OCRResult(
                        engine_name=self.name,
                        pdf_name=pdf_path.stem,
                        page_number=None,
                        text_plain="",
                        text_markdown="",
                        metadata={},
                        success=False,
                        error_message=str(e),
                    )
                ]

        logger.info(f"[{self.name}] Processing PDF: {pdf_path.name}")
        results: List[OCRResult] = []

        try:
            with Timer() as t:
                conv_result = self._converter.convert(str(pdf_path))

            doc = conv_result.document

            # Export full document as markdown and plain text
            full_markdown = doc.export_to_markdown()
            full_plain = doc.export_to_text()

            # Split by page if page information is available
            pages = getattr(doc, "pages", None)
            if pages:
                for page in pages:
                    page_num = page.page_no + 1  # docling uses 0-based index
                    page_md = _extract_page_markdown(doc, page.page_no)
                    page_plain = _extract_page_text(doc, page.page_no)
                    results.append(
                        OCRResult(
                            engine_name=self.name,
                            pdf_name=pdf_path.stem,
                            page_number=page_num,
                            text_plain=page_plain,
                            text_markdown=page_md,
                            metadata={
                                "use_gpu": self.use_gpu,
                                "total_pages": len(pages),
                                "chars_plain": len(page_plain),
                                "chars_markdown": len(page_md),
                            },
                            success=True,
                            execution_time=t.duration / max(len(pages), 1),
                        )
                    )
            else:
                # Fallback: single result for the whole document
                results.append(
                    OCRResult(
                        engine_name=self.name,
                        pdf_name=pdf_path.stem,
                        page_number=None,
                        text_plain=full_plain,
                        text_markdown=full_markdown,
                        metadata={
                            "use_gpu": self.use_gpu,
                            "chars_plain": len(full_plain),
                            "chars_markdown": len(full_markdown),
                        },
                        success=True,
                        execution_time=t.duration,
                    )
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error processing {pdf_path.name}: {e}")
            results.append(
                OCRResult(
                    engine_name=self.name,
                    pdf_name=pdf_path.stem,
                    page_number=None,
                    text_plain="",
                    text_markdown="",
                    metadata={"use_gpu": self.use_gpu},
                    success=False,
                    error_message=str(e),
                )
            )

        return results

    def process_page(self, image_path: Path, page_number: int, pdf_name: str) -> OCRResult:
        # Docling works at PDF level; page-level image processing is not supported
        return OCRResult(
            engine_name=self.name,
            pdf_name=pdf_name,
            page_number=page_number,
            text_plain="",
            text_markdown="",
            metadata={},
            success=False,
            error_message="Docling processes full PDFs, not individual page images",
        )


# ---------------------------------------------------------------------------
# Helpers to extract per-page content from a Docling document
# ---------------------------------------------------------------------------

def _extract_page_markdown(doc, page_no: int) -> str:
    """Return markdown text for a specific 0-based page number."""
    lines = []
    for element, _ in doc.iterate_items():
        prov = getattr(element, "prov", None)
        if prov and any(p.page_no == page_no for p in prov):
            try:
                lines.append(element.export_to_markdown())
            except Exception:
                pass
    return "\n\n".join(lines)


def _extract_page_text(doc, page_no: int) -> str:
    """Return plain text for a specific 0-based page number."""
    lines = []
    for element, _ in doc.iterate_items():
        prov = getattr(element, "prov", None)
        if prov and any(p.page_no == page_no for p in prov):
            text = getattr(element, "text", None)
            if text:
                lines.append(text)
    return "\n".join(lines)

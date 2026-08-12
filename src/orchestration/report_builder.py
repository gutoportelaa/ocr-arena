import pandas as pd
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ReportBuilder:
    def __init__(self, reports_dir: str):
        self.reports_dir = Path(reports_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_markdown_report(
        self, summaries: Dict[str, Any], routing_rules: str
    ) -> str:
        logger.info("Building final markdown report")

        sections = [
            "# OCR Benchmark — Relatório Comparativo\n",
            self._section_overview(summaries),
            self._section_timing(summaries),
            self._section_chars(summaries),
            self._section_docling_speedup(summaries),
            self._section_by_pdf(summaries),
            self._section_routing(routing_rules),
            self._section_methodology(),
        ]

        return "\n".join(sections)

    def save_report(self, content: str, filename: str = "final_report.md"):
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        path = self.reports_dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info(f"Report saved to {path}")

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    def _section_overview(self, summaries: Dict[str, Any]) -> str:
        by_engine: pd.DataFrame = summaries.get("by_engine", pd.DataFrame())
        if by_engine.empty:
            return "## 1. Resumo por Engine\n\n_Nenhum dado disponível._\n"

        cols = [
            "success_rate_%",
            "avg_time_per_page",
            "total_time",
            "avg_chars_extracted",
            "total_chars_extracted",
            "pages_processed",
            "pdfs_processed",
        ]
        cols = [c for c in cols if c in by_engine.columns]

        lines = [
            "## 1. Resumo por Engine\n",
            "Engines ordenados por tempo médio por página (crescente).\n",
            by_engine[cols].to_markdown(),
            "\n",
            "**Legenda:**",
            "- `success_rate_%` — % de páginas processadas com sucesso",
            "- `avg_time_per_page` — tempo médio por página (segundos)",
            "- `total_time` — tempo total acumulado (segundos)",
            "- `avg_chars_extracted` — média de caracteres extraídos por página",
            "- `total_chars_extracted` — total de caracteres extraídos",
            "\n",
        ]
        return "\n".join(lines)

    def _section_timing(self, summaries: Dict[str, Any]) -> str:
        timing: pd.DataFrame = summaries.get("timing_comparison", pd.DataFrame())
        if timing.empty:
            return "## 2. Comparativo de Tempo por Documento\n\n_Nenhum dado disponível._\n"

        lines = [
            "## 2. Comparativo de Tempo por Documento (segundos)\n",
            "Tempo total por engine para cada documento do benchmark.\n",
            timing.to_markdown(),
            "\n",
        ]
        return "\n".join(lines)

    def _section_chars(self, summaries: Dict[str, Any]) -> str:
        chars: pd.DataFrame = summaries.get("chars_comparison", pd.DataFrame())
        if chars.empty:
            return "## 3. Comparativo de Caracteres Extraídos por Documento\n\n_Nenhum dado disponível._\n"

        lines = [
            "## 3. Comparativo de Caracteres Extraídos por Documento\n",
            "Quantidade de caracteres extraídos por engine para cada documento. "
            "Volumes maiores indicam maior cobertura textual (não necessariamente maior qualidade).\n",
            chars.to_markdown(),
            "\n",
        ]
        return "\n".join(lines)

    def _section_docling_speedup(self, summaries: Dict[str, Any]) -> str:
        speedup = summaries.get("docling_speedup")
        if not speedup:
            return (
                "## 4. Docling: CPU vs GPU\n\n"
                "_Speedup não calculado — engine `docling` ou `docling_gpu` não disponível nos resultados._\n"
            )

        cpu_t = speedup["docling_cpu_total_s"]
        gpu_t = speedup["docling_gpu_total_s"]
        factor = speedup["gpu_speedup_factor"]

        lines = [
            "## 4. Docling: CPU vs GPU\n",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Tempo total — Docling CPU (s) | `{cpu_t}` |",
            f"| Tempo total — Docling GPU/CUDA (s) | `{gpu_t}` |",
            f"| Fator de speedup GPU/CPU | **{factor}×** |",
            "",
            "> O speedup real pode variar conforme o tamanho dos documentos, "
            "a GPU disponível e a carga do sistema.",
            "\n",
        ]
        return "\n".join(lines)

    def _section_by_pdf(self, summaries: Dict[str, Any]) -> str:
        by_pdf: pd.DataFrame = summaries.get("by_pdf", pd.DataFrame())
        if by_pdf.empty:
            return "## 5. Desempenho por Documento\n\n_Nenhum dado disponível._\n"

        lines = [
            "## 5. Desempenho por Documento\n",
            by_pdf.head(40).to_markdown(),
            "\n",
        ]
        return "\n".join(lines)

    def _section_routing(self, routing_rules: str) -> str:
        return (
            "## 6. Pipeline de Roteamento Recomendado\n\n"
            + routing_rules.strip()
            + "\n\n"
        )

    def _section_methodology(self) -> str:
        return (
            "## 7. Metodologia\n\n"
            "- **Seleção:** 20 PDFs escolhidos por score de complexidade "
            "(alta, média, baixa, outliers).\n"
            "- **Engines avaliados:** Tesseract, Docling (CPU), Docling (GPU/CUDA), "
            "Marker, MarkItDown, Ollama (olmocr2, glm-ocr, minicpm-v).\n"
            "- **Hardware:** execução local — GPU NVIDIA RTX 4070 para engines com aceleração.\n"
            "- **Métricas:** taxa de sucesso, tempo de execução por página, "
            "total de caracteres extraídos.\n"
            "- **Resolução de renderização:** 300 DPI.\n"
            "- **Seed aleatório:** 42.\n"
        )

"""ETF report capability: writes run summaries to JSON and Markdown files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mlblack.core.capability import Capability


@dataclass(frozen=True)
class EtfReportSpec:
    """Report output specification for the ETF case."""

    output_dir: str = "runs/etf_temporal_forecast"
    run_id: str = "etf_temporal_forecast"
    write_json: bool = True
    write_markdown: bool = True


class EtfReportPlugin(Capability):
    """Persist ETF temporal forecast run summaries."""

    name = "etf_report"
    context_requires = ()
    context_optional = ("trainer.context", "trainer.snapshot_store")
    context_provides = ("plugin.etf_report_path",)
    context_mutates = ("trainer.context",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Writes ETF run report artifacts as JSON/Markdown."

    def __init__(
        self,
        *,
        output_dir: str | Path = "runs/etf_temporal_forecast",
        run_id: str = "etf_temporal_forecast",
        write_json: bool = True,
        write_markdown: bool = True,
    ) -> None:
        self.spec = EtfReportSpec(
            output_dir=str(output_dir),
            run_id=str(run_id),
            write_json=bool(write_json),
            write_markdown=bool(write_markdown),
        )

    def on_fit_end(self, trainer: Any, context: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        output_dir = Path(self.spec.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = dict(report or {})
        payload = {
            "run_id": self.spec.run_id,
            "summary": summary,
            "context": dict(context or {}),
        }

        if self.spec.write_json:
            json_path = output_dir / f"{self.spec.run_id}.etf_report.json"
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

        if self.spec.write_markdown:
            md_path = output_dir / f"{self.spec.run_id}.etf_report.md"
            md_path.write_text(self._render_markdown(summary), encoding="utf-8")

        if hasattr(trainer, "context") and isinstance(getattr(trainer, "context"), dict):
            trainer.context_store["plugin.etf_report_path"] = str(output_dir)

    def _render_markdown(self, summary: Mapping[str, Any]) -> str:
        aggregate = dict(summary.get("aggregate", {}) or {})
        dataset = dict(summary.get("dataset", {}) or {})

        lines = [
            "# ETF Temporal Forecast Report",
            "",
            f"- Run ID: `{self.spec.run_id}`",
            f"- Dataset: `{dataset.get('label', 'unknown')}`",
            f"- Rows: `{dataset.get('rows', 'n/a')}`",
            f"- Assets: `{dataset.get('assets', 'n/a')}`",
            f"- Date Range: `{dataset.get('start', 'n/a')}` .. `{dataset.get('end', 'n/a')}`",
            "",
            "## Aggregate Metrics",
        ]

        metric_rows = [
            ("Test RMSE", aggregate.get("composite_test_rmse_mean")),
            ("Direction Accuracy", aggregate.get("composite_direction_accuracy_mean")),
            ("Rank IC Mean", aggregate.get("composite_rank_ic_mean")),
            ("Rank IC Std", aggregate.get("composite_rank_ic_std")),
            ("Hit Rate", aggregate.get("composite_hit_rate_mean")),
            ("Net Sharpe Proxy", aggregate.get("composite_net_sharpe_proxy_mean")),
            ("Max Drawdown Abs", aggregate.get("composite_max_drawdown_abs_mean")),
            ("Turnover Proxy", aggregate.get("composite_turnover_proxy_mean")),
        ]
        for label, value in metric_rows:
            lines.append(f"- {label}: `{value}`")

        return "\n".join(lines) + "\n"

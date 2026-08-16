"""ETF observability capability: writes a compact run trace manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mlblack.core.capability import Capability


@dataclass(frozen=True)
class EtfObservabilitySpec:
    """Observability output specification for the ETF case."""

    output_dir: str = "runs/etf_temporal_forecast"
    run_id: str = "etf_temporal_forecast"
    write_json: bool = True
    write_markdown: bool = True


class EtfObservabilityPlugin(Capability):
    """Persist a compact observability manifest for ETF case runs."""

    name = "etf_observability"
    context_requires = ()
    context_optional = ("trainer.context", "trainer.snapshot_store")
    context_provides = ("plugin.etf_observability_path",)
    context_mutates = ("trainer.context",)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Writes a compact ETF observability manifest as JSON/Markdown."

    def __init__(
        self,
        *,
        output_dir: str | Path = "runs/etf_temporal_forecast",
        run_id: str = "etf_temporal_forecast",
        write_json: bool = True,
        write_markdown: bool = True,
    ) -> None:
        self.spec = EtfObservabilitySpec(
            output_dir=str(output_dir),
            run_id=str(run_id),
            write_json=bool(write_json),
            write_markdown=bool(write_markdown),
        )

    def on_fit_end(self, trainer: Any, context: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        output_dir = Path(self.spec.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = dict(report or {})
        aggregate = dict(summary.get("aggregate", {}) or {})
        dataset = dict(summary.get("dataset", {}) or {})
        manifest = {
            "run_id": self.spec.run_id,
            "suite_id": summary.get("suite_id", self.spec.run_id),
            "case": summary.get("case", "etf_temporal_forecast"),
            "fold_count": summary.get("fold_count"),
            "dataset": dataset,
            "aggregate": aggregate,
            "context": dict(context or {}),
            "artifacts": {
                "report_json": f"{self.spec.run_id}.etf_report.json",
                "report_md": f"{self.spec.run_id}.etf_report.md",
                "observability_json": f"{self.spec.run_id}.observability.json",
                "observability_md": f"{self.spec.run_id}.observability.md",
            },
        }

        if self.spec.write_json:
            json_path = output_dir / f"{self.spec.run_id}.observability.json"
            json_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )

        if self.spec.write_markdown:
            md_path = output_dir / f"{self.spec.run_id}.observability.md"
            md_path.write_text(self._render_markdown(manifest), encoding="utf-8")

        if hasattr(trainer, "context") and isinstance(getattr(trainer, "context"), dict):
            trainer.context["plugin.etf_observability_path"] = str(output_dir)

    def _render_markdown(self, manifest: Mapping[str, Any]) -> str:
        dataset = dict(manifest.get("dataset", {}) or {})
        aggregate = dict(manifest.get("aggregate", {}) or {})

        lines = [
            "# ETF Temporal Forecast Observability",
            "",
            f"- Run ID: `{self.spec.run_id}`",
            f"- Suite ID: `{manifest.get('suite_id', self.spec.run_id)}`",
            f"- Case: `{manifest.get('case', 'etf_temporal_forecast')}`",
            f"- Folds: `{manifest.get('fold_count', 'n/a')}`",
            f"- Dataset: `{dataset.get('label', 'unknown')}`",
            f"- Rows: `{dataset.get('rows', 'n/a')}`",
            f"- Assets: `{dataset.get('assets', 'n/a')}`",
            f"- Date Range: `{dataset.get('start', 'n/a')}` .. `{dataset.get('end', 'n/a')}`",
            "",
            "## Core Metrics",
            f"- RMSE: `{aggregate.get('composite_test_rmse_mean')}`",
            f"- Direction Accuracy: `{aggregate.get('composite_direction_accuracy_mean')}`",
            f"- Rank IC Mean: `{aggregate.get('composite_rank_ic_mean')}`",
            f"- Rank IC Std: `{aggregate.get('composite_rank_ic_std')}`",
            f"- Hit Rate: `{aggregate.get('composite_hit_rate_mean')}`",
            f"- Net Sharpe Proxy: `{aggregate.get('composite_net_sharpe_proxy_mean')}`",
            f"- Max Drawdown Abs: `{aggregate.get('composite_max_drawdown_abs_mean')}`",
            f"- Turnover Proxy: `{aggregate.get('composite_turnover_proxy_mean')}`",
        ]
        return "\n".join(lines) + "\n"

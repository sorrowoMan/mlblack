from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from core.orchestration.capabilities import FlowCapability
from nowcasting_work_ci.mlblack_side.runtime.contracts import RuntimeContextKey, SummaryReportPayload, ctx_set

from .report_writer import write_summary_report


@dataclass
class ReportWriterPlugin(FlowCapability):
    """Runtime hook that writes final summary artifacts on experiment finish."""

    name: str = "report_writer"
    priority: int = 200
    enabled: bool = True
    is_algorithmic: bool = False
    config: dict[str, Any] = field(default_factory=dict)
    context_requires: Sequence[str] = ("report_payload",)
    context_provides: Sequence[str] = (RuntimeContextKey.SUMMARY_PATH.value,)
    context_mutates: Sequence[str] = (RuntimeContextKey.SUMMARY_PATH.value,)
    context_cache: Sequence[str] = tuple()
    context_notes: str | None = "Writes final summary artifacts on experiment finish."
    strict: bool = False
    payload_key: str = "report_payload"

    def on_experiment_start(self, context: Mapping[str, Any]) -> None:
        return

    def on_stage_start(self, stage: str, context: Mapping[str, Any]) -> None:
        return

    def on_stage_end(self, stage: str, payload: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        return

    def on_stage_error(self, stage: str, error: Exception, context: Mapping[str, Any]) -> None:
        return

    def on_experiment_finish(self, result: Any, context: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        if not isinstance(result, Mapping):
            return
        payload = result.get(self.payload_key)
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            self._maybe_raise(TypeError(f"result['{self.payload_key}'] must be a mapping"))
            return
        try:
            normalized = SummaryReportPayload.from_mapping(payload)
        except Exception as exc:
            self._maybe_raise(exc)
            return

        report_path = write_summary_report(
            report=normalized.report,
            out_root=normalized.out_root,
            sym_rmse=float(normalized.sym_rmse),
            xgb_rmse=float(normalized.xgb_rmse),
            sym_interval=normalized.sym_interval,
            xgb_interval=normalized.xgb_interval,
            interval_alpha=float(normalized.interval_alpha),
        )
        if isinstance(context, dict):
            ctx_set(context, RuntimeContextKey.SUMMARY_PATH, str(report_path))

    def on_experiment_error(self, error: Exception, context: Mapping[str, Any]) -> None:
        return

    def _maybe_raise(self, error: Exception) -> None:
        if self.strict:
            raise error


__all__ = ["ReportWriterPlugin"]

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    from _bootstrap import ensure_case_importable  # noqa: E402
else:
    from ._bootstrap import ensure_case_importable  # noqa: E402

ensure_case_importable(Path(__file__))

from legacy_nowcasting.legacy_runner import check_payload, run_legacy_case  # noqa: E402


@dataclass
class SymbolicIntervalOuterBridge:
    """Standard Case bridge for the historical Work-CI native interval script."""

    resource_context: Mapping[str, Any] = field(default_factory=dict)
    default_args: Sequence[str] = ()
    case_kind: str = "interval"

    @property
    def problem(self) -> str:
        return "traffic_symbolic_interval_forecasting"

    @property
    def adapter(self) -> str:
        return "nsgablack_outer_search_bridge"

    @property
    def pipeline(self) -> str:
        return "legacy_nowcasting_native_interval_pipeline"

    def check(self) -> dict[str, Any]:
        return check_payload(self.case_kind, self.default_args).as_dict()

    def run(self, argv: Sequence[str] = ()) -> int:
        return run_legacy_case(self.case_kind, [*self.default_args, *tuple(argv)])


def build_solver(
    *,
    resource_context: Mapping[str, Any] | None = None,
    default_args: Sequence[str] = (),
    component_overrides: Mapping[str, Any] | None = None,
) -> SymbolicIntervalOuterBridge:
    del component_overrides
    return SymbolicIntervalOuterBridge(
        resource_context=dict(resource_context or {}),
        default_args=tuple(str(item) for item in default_args),
    )


__all__ = ["SymbolicIntervalOuterBridge", "build_solver"]

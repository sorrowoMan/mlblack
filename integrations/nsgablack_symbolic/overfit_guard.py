from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class OverfitGuardConfig:
    enabled: bool = True
    max_generalization_gap: float = 0.25
    max_valid_train_ratio: float = 2.5
    min_valid_score: float | None = None
    penalty_weight: float = 1.0
    metric_name: str = "rmse"
    lower_is_better: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OverfitGuardReport:
    triggered: bool
    penalty: float
    reasons: tuple[str, ...]
    train_score: float | None
    valid_score: float | None
    gap: float | None
    ratio: float | None
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": bool(self.triggered),
            "penalty": float(self.penalty),
            "reasons": list(self.reasons),
            "train_score": self.train_score,
            "valid_score": self.valid_score,
            "gap": self.gap,
            "ratio": self.ratio,
            "config": dict(self.config),
        }


class SymbolicOverfitGuard:
    """Scores train/validation drift for symbolic structure search."""

    name = "symbolic_overfit_guard"
    context_requires = ("feedback.metrics",)
    context_optional = ("data.X_valid", "data.y_valid", "symbolic.candidate_score")
    context_provides = ("symbolic.overfit_guard",)
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Produces a lightweight overfit penalty from train/valid metrics."

    def __init__(self, config: OverfitGuardConfig | None = None) -> None:
        self.config = config or OverfitGuardConfig()

    def evaluate(
        self,
        metrics: Mapping[str, Any],
        *,
        metric_name: str | None = None,
        train_prefix: str = "train",
        valid_prefix: str = "valid",
    ) -> OverfitGuardReport:
        cfg = self.config
        if not bool(cfg.enabled):
            return self._report(False, 0.0, (), None, None, None, None)
        metric = str(metric_name or cfg.metric_name)
        train = _float_or_none(metrics.get(f"{train_prefix}.{metric}"))
        valid = _float_or_none(metrics.get(f"{valid_prefix}.{metric}"))
        if train is None or valid is None:
            return self._report(False, 0.0, ("missing_train_or_valid_metric",), train, valid, None, None)

        if bool(cfg.lower_is_better):
            gap = float(valid - train)
            ratio = float(valid / max(abs(train), 1e-12))
            min_valid_trigger = cfg.min_valid_score is not None and valid > float(cfg.min_valid_score)
        else:
            gap = float(train - valid)
            ratio = float(train / max(abs(valid), 1e-12))
            min_valid_trigger = cfg.min_valid_score is not None and valid < float(cfg.min_valid_score)

        reasons: list[str] = []
        if gap > float(cfg.max_generalization_gap):
            reasons.append("generalization_gap")
        if ratio > float(cfg.max_valid_train_ratio):
            reasons.append("valid_train_ratio")
        if min_valid_trigger:
            reasons.append("valid_score_floor")

        raw_penalty = 0.0
        if "generalization_gap" in reasons:
            raw_penalty += max(0.0, gap - float(cfg.max_generalization_gap))
        if "valid_train_ratio" in reasons:
            raw_penalty += max(0.0, ratio - float(cfg.max_valid_train_ratio))
        if "valid_score_floor" in reasons and cfg.min_valid_score is not None:
            raw_penalty += abs(float(valid) - float(cfg.min_valid_score))
        penalty = float(cfg.penalty_weight) * float(raw_penalty)
        return self._report(bool(reasons), penalty, tuple(reasons), train, valid, gap, ratio)

    def _report(
        self,
        triggered: bool,
        penalty: float,
        reasons: tuple[str, ...],
        train: float | None,
        valid: float | None,
        gap: float | None,
        ratio: float | None,
    ) -> OverfitGuardReport:
        cfg = self.config
        return OverfitGuardReport(
            triggered=bool(triggered),
            penalty=float(penalty),
            reasons=tuple(reasons),
            train_score=train,
            valid_score=valid,
            gap=gap,
            ratio=ratio,
            config={
                "enabled": bool(cfg.enabled),
                "max_generalization_gap": float(cfg.max_generalization_gap),
                "max_valid_train_ratio": float(cfg.max_valid_train_ratio),
                "min_valid_score": cfg.min_valid_score,
                "penalty_weight": float(cfg.penalty_weight),
                "metric_name": str(cfg.metric_name),
                "lower_is_better": bool(cfg.lower_is_better),
                "metadata": dict(cfg.metadata),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self._report(False, 0.0, (), None, None, None, None).config}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


__all__ = ["OverfitGuardConfig", "OverfitGuardReport", "SymbolicOverfitGuard"]

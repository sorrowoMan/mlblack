from __future__ import annotations


def coverage_error(*, picp: float, coverage_target: float) -> float:
    return float(abs(float(picp) - float(coverage_target)))


def interval_objective_sort_key(
    *,
    coverage_error_value: float,
    pinaw: float,
    interval_score: float,
    coverage_error_threshold: float,
) -> tuple[float, float, float, float]:
    cov = float(coverage_error_value)
    pin = float(pinaw)
    iscore = float(interval_score)
    thr = float(coverage_error_threshold)
    if cov <= thr:
        return (0.0, pin, iscore, cov)
    return (1.0, cov, pin, iscore)


__all__ = ["coverage_error", "interval_objective_sort_key"]

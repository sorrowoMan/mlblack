"""ML capability lifecycle mapped onto the shared blackbase Plugin API."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from blackbase.plugin import PluginBase


def _trainer_context(trainer: Any) -> dict[str, Any]:
    if trainer is None:
        return {}
    builder = getattr(trainer, "build_context", None)
    if callable(builder):
        value = builder()
        if isinstance(value, Mapping):
            return dict(value)
    store = getattr(trainer, "context_store", None)
    snapshot = getattr(store, "snapshot", None)
    if callable(snapshot):
        value = snapshot()
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(store, Mapping):
        return dict(store)
    return {}


def _step_row(trainer: Any, generation: int) -> dict[str, Any]:
    history = getattr(trainer, "history", None) if trainer is not None else None
    if history:
        latest = history[-1]
        if isinstance(latest, Mapping):
            return dict(latest)
    return {"step": int(generation)}


class Capability(PluginBase):
    """ML semantic Plugin with fit-oriented hook names."""

    def __init__(self, name: Optional[str] = None, priority: int = 0, **kwargs):
        super().__init__(name=name or "capability", priority=priority, **kwargs)

    def on_fit_start(self, trainer, context):
        return None

    def on_step_start(self, trainer, context, row):
        return None

    def on_evaluate_start_ml(self, trainer, candidate, context):
        return None

    def on_evaluate_end_ml(self, trainer, candidate, feedback, context):
        return None

    def on_step_end(self, trainer, context, row):
        return None

    def on_fit_end(self, trainer, context, result):
        return None

    def on_error_ml(self, trainer, error, context):
        return None

    def on_solver_init(self, solver):
        self.on_fit_start(solver, _trainer_context(solver))

    def on_population_init(self, population, objectives, violations):
        return None

    def on_generation_start(self, generation):
        self.on_step_start(self.solver, _trainer_context(self.solver), {"step": int(generation)})

    def on_evaluate_start(self, candidate, context=None):
        self.on_evaluate_start_ml(self.solver, candidate, dict(context or {}))

    def on_evaluate_end(self, candidate, feedback, context=None):
        self.on_evaluate_end_ml(self.solver, candidate, feedback, dict(context or {}))

    def on_generation_end(self, generation):
        self.on_step_end(
            self.solver,
            _trainer_context(self.solver),
            _step_row(self.solver, generation),
        )

    def on_solver_finish(self, result):
        report = result.get("report", {}) if isinstance(result, Mapping) else result
        if not isinstance(report, Mapping):
            report = {"result": report}
        self.on_fit_end(self.solver, _trainer_context(self.solver), dict(report))

    def on_error(self, error, context=None):
        self.on_error_ml(self.solver, error, dict(context or {}))

__all__ = ["Capability"]

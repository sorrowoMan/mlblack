from __future__ import annotations

from typing import Any, Mapping

from .contracts import ComponentContract, ContractMixin


class Capability(ContractMixin):
    """Lifecycle capability, equivalent to nsgablack Plugin."""

    name = "capability"
    context_requires = ()
    context_optional = ('trainer.context', 'trainer.snapshot_store')
    context_provides = ('capability.side_effect',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides capability.side_effect; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("trainer.context", "trainer.snapshot_store"),
        provides=("capability.side_effect",),
        mutates=("trainer.context",),
    )

    def on_fit_start(self, trainer: Any, context: Mapping[str, Any]) -> None:
        return None

    def on_step_start(self, trainer: Any, context: Mapping[str, Any]) -> None:
        return None

    def on_evaluate_start(self, trainer: Any, candidate: Any, context: Mapping[str, Any]) -> None:
        return None

    def on_evaluate_end(self, trainer: Any, candidate: Any, feedback: Any, context: Mapping[str, Any]) -> None:
        return None

    def on_step_end(self, trainer: Any, context: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        return None

    def on_fit_end(self, trainer: Any, context: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        return None

    def on_error(self, trainer: Any, error: BaseException, context: Mapping[str, Any]) -> None:
        return None

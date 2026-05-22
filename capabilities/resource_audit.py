from __future__ import annotations

from typing import Any, Mapping

from mlblack.core.capability import Capability
from mlblack.core.contracts import ComponentContract


class ResourceAuditCapability(Capability):
    name = "resource_audit"
    context_requires = ()
    context_optional = ('resource_context', 'trainer.context')
    context_provides = ('resource.audit',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides resource.audit; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("resource_context", "trainer.context"),
        provides=("resource.audit",),
        mutates=("trainer.context",),
        metadata={"capability": "resource_audit"},
    )

    def on_fit_start(self, trainer: Any, context: Mapping[str, Any]) -> None:
        trainer.context_store["resource.audit.fit_start"] = dict(context.get("resource_context", {}) or {})

    def on_step_end(self, trainer: Any, context: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        trainer.context_store["resource.audit.last_step"] = {
            "step": int(row.get("step", 0)),
            "resource_context": dict(context.get("resource_context", {}) or {}),
        }

    def on_fit_end(self, trainer: Any, context: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        trainer.context_store["resource.audit.fit_end"] = {
            "resource_context": dict(context.get("resource_context", {}) or {}),
            "report_resources": dict(report.get("resources", {}) or {}),
        }


from __future__ import annotations

from blackbase.plugin import Plugin

from mlblack.capabilities.resource_audit import ResourceAuditCapability
from mlblack.core.trainer import BlankTrainer


def test_blank_trainer_build_context_runs_shared_plugin_chain() -> None:
    class ContextPlugin(Plugin):
        def __init__(self) -> None:
            super().__init__("context_plugin")

        def on_context_build(self, context):
            return {**context, "plugin.context": "seen"}

    trainer = BlankTrainer(run_name="context_demo")
    trainer.add_plugin(ContextPlugin())

    context = trainer.build_context()

    assert context["run_name"] == "context_demo"
    assert context["plugin.context"] == "seen"


def test_legacy_capability_uses_shared_blackbase_adapter() -> None:
    trainer = BlankTrainer(
        run_name="resource_demo",
        resource_context={"threads": 2, "namespace": "resource_demo"},
    )
    capability = ResourceAuditCapability()
    trainer.add_capability(capability)

    trainer.plugin_manager.on_solver_init(trainer)
    trainer.history.append({"step": 4, "score": 0.25})
    trainer.plugin_manager.on_generation_end(4)
    trainer.plugin_manager.on_solver_finish(
        {"report": {"resources": trainer.resource_context.as_dict()}}
    )

    assert trainer.context_store["resource.audit.fit_start"]["threads"] == 2
    assert trainer.context_store["resource.audit.last_step"]["step"] == 4
    assert trainer.context_store["resource.audit.fit_end"]["report_resources"]["threads"] == 2
    assert type(trainer._capability_adapters[0]).__module__.startswith(
        "blackbase.adapters.mlblack.plugin"
    )

from __future__ import annotations

from blackbase.plugin import PluginBase, PluginManager

from mlblack.capabilities.resource_audit import ResourceAuditCapability
from mlblack.core.trainer import BlankTrainer
from mlblack.core.capability import Capability


def test_blank_trainer_build_context_runs_shared_plugin_chain() -> None:
    class ContextPlugin(PluginBase):
        def __init__(self) -> None:
            super().__init__("context_plugin")

        def on_context_build(self, context):
            return {**context, "plugin.context": "seen"}

    trainer = BlankTrainer(run_name="context_demo")
    trainer.add_plugin(ContextPlugin())

    context = trainer.build_context()

    assert context["run_name"] == "context_demo"
    assert context["plugin.context"] == "seen"


def test_ml_capability_uses_shared_plugin_lifecycle_directly() -> None:
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
    assert trainer.plugin_manager.get(capability.name) is capability


def test_capability_preserves_trainer_context_and_rows() -> None:
    class ContextCapability(Capability):
        context_provides = ("ml.seen",)

        def __init__(self) -> None:
            super().__init__(name="context_capability")
            self.events = []

        def on_fit_start(self, trainer, context):
            self.events.append(("fit_start", trainer, dict(context)))

        def on_step_end(self, trainer, context, row):
            self.events.append(("step_end", trainer, dict(context), dict(row)))

    trainer = BlankTrainer(run_name="adapter-context")
    trainer.history.append({"step": 3, "score": 0.5})
    capability = ContextCapability()
    manager = PluginManager(strict=True)
    manager.register(capability)
    capability.attach(trainer)

    manager.on_solver_init(trainer)
    manager.on_generation_end(3)

    assert capability.events[0][2]["run_name"] == "adapter-context"
    assert capability.events[1][3] == {"step": 3, "score": 0.5}
    assert capability.get_context_contract()["provides"] == ("ml.seen",)

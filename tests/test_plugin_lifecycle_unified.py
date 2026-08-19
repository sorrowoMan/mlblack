from __future__ import annotations

from blackbase.plugin import PluginBase

from mlblack.core import Capability, Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter


class _Representation(ModelRepresentation):
    def init(self, context):
        del context
        return UnknownState([0.0])

    def decode(self, state, context=None):
        del context
        return state.as_array()


class _Problem(LearningProblem):
    def evaluate(self, model, state, context=None):
        del model, state, context
        return Feedback(objectives=[0.0])


def _solver():
    return build_learning_solver(
        problem=_Problem(),
        representation=_Representation(),
        adapter=FixedCandidateAdapter(),
        run_name="plugin_demo",
    )


def test_learning_solver_build_context_runs_shared_plugin_chain() -> None:
    class ContextPlugin(PluginBase):
        def __init__(self) -> None:
            super().__init__("context_plugin")

        def on_context_build(self, context):
            return {**context, "plugin.context": "seen"}

    solver = _solver()
    solver.add_plugin(ContextPlugin())
    assert solver.build_context()["plugin.context"] == "seen"


def test_ml_capability_maps_fit_words_to_shared_solver_lifecycle() -> None:
    class ContextCapability(Capability):
        context_provides = ("ml.seen",)

        def __init__(self) -> None:
            super().__init__(name="context_capability")
            self.events = []

        def on_fit_start(self, trainer, context):
            self.events.append(("fit_start", trainer, dict(context)))

        def on_step_end(self, trainer, context, row):
            self.events.append(("step_end", trainer, dict(context), dict(row)))

    solver = _solver()
    capability = ContextCapability()
    solver.add_capability(capability)
    solver.fit(max_steps=1)

    assert capability.events[0][0] == "fit_start"
    assert capability.events[0][1] is solver
    assert capability.events[0][2]["run_name"] == "plugin_demo"
    assert any(event[0] == "step_end" for event in capability.events)
    assert capability.get_context_contract()["provides"] == ("ml.seen",)

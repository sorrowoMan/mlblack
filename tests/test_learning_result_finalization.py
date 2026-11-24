from __future__ import annotations
from blackbase.participation import ParticipationContract

import numpy as np
import pytest

from blackbase import CandidateBatch
from blackbase.resources import DataRef
from mlblack.core import Feedback, LearningProblem, ModelRepresentation, UnknownState
from mlblack.integrations import build_learning_solver
from nsgablack.adapters import FixedCandidateAdapter
from nsgablack.core.state.incumbent import IncumbentState
from nsgablack.core.evaluation_feedback import OptimizationFeedbackBatch
from nsgablack.core.state.incumbent import CandidateProvenance
from nsgablack.plugins import Plugin


class _Problem(LearningProblem):
    def evaluate(self, model, state, context=None):
        del model, state, context
        return Feedback(objectives=[0.0], metrics={"loss": 0.0})


class _NumericEquivalentRepresentation(ModelRepresentation):
    def __init__(self, size: int = 1) -> None:
        self.size = int(size)

    def init(self, context):
        del context
        return UnknownState(np.zeros(self.size), metadata={"architecture": "initial"})

    def decode(self, state, context=None):
        del context
        return state.as_array()

    def equivalent(self, left, right):
        return np.array_equal(left.as_array(), right.as_array())


class _CaseRuntime:
    def __init__(self) -> None:
        self.artifact_refs: dict[str, DataRef] = {}
        self._finalization_transaction = None
        self._finalization_observers = []

    @property
    def finalization_artifact_refs(self):
        transaction = self._finalization_transaction
        return {} if transaction is None else dict(transaction.staged)

    def checkpoint(self) -> None:
        return None

    def _materialize_artifact(self, name, value, **kwargs):
        assert value is not None
        return DataRef(
            uri=f"memory://learning-result/{name}",
            kind=str(kwargs.get("kind", "artifact")),
            backend="memory",
        )

    def publish_artifact(self, name, value, **kwargs):
        ref = self._materialize_artifact(name, value, **kwargs)
        self.artifact_refs[str(name)] = ref
        return ref

    def begin_artifact_transaction(self, label):
        del label
        runtime = self

        class _Transaction:
            def __init__(self):
                self.staged: dict[str, DataRef] = {}

            def publish(self, name, value, **kwargs):
                ref = runtime._materialize_artifact(name, value, **kwargs)
                self.staged[str(name)] = ref
                return ref

            def commit(self):
                runtime.artifact_refs.update(self.staged)
                return dict(self.staged)

            def prepare(self):
                return dict(self.staged)

            def abort(self, reason="aborted"):
                del reason
                self.staged.clear()

        return _Transaction()

    def begin_finalization_transaction(self, label):
        if self._finalization_transaction is None:
            self._finalization_transaction = self.begin_artifact_transaction(label)
        return self._finalization_transaction

    def abort_finalization_transaction(self, reason="aborted"):
        if self._finalization_transaction is not None:
            self._finalization_transaction.abort(reason)

    def register_finalization_observer(self, observer, *, name="observer"):
        self._finalization_observers.append((str(name), observer))

    def seal_finalization(self):
        if self._finalization_transaction is not None:
            self.artifact_refs.update(self._finalization_transaction.prepare())
        failures = []
        for name, observer in tuple(self._finalization_observers):
            try:
                observer({})
            except BaseException as exc:
                failures.append((name, exc))
        return tuple(failures)


def _solver(
    *,
    size: int = 1,
    inline_limit: int = 64 * 1024,
    plugin_strict: bool = False,
):
    return build_learning_solver(
        problem=_Problem(),
        representation=_NumericEquivalentRepresentation(size),
        adapter=FixedCandidateAdapter(),
        run_name="result_finalization",
        result_inline_limit_bytes=inline_limit,
        plugin_strict=plugin_strict,
    )


def test_incumbent_token_has_priority_over_numeric_equivalence() -> None:
    solver = _solver()
    first = UnknownState([1.0], metadata={"architecture": "first"})
    second = UnknownState([1.0], metadata={"architecture": "second"})
    first_feedback = Feedback(objectives=[10.0], metrics={"source": "first"})
    second_feedback = Feedback(objectives=[1.0], metrics={"source": "second"})
    solver._record_ml_model_handle(
        "first-model",
        candidate_token="candidate:first",
    )
    solver._record_ml_model_handle(
        "second-model",
        candidate_token="candidate:second",
    )
    batch = CandidateBatch(
        semantic_states=(first, second),
        numeric_matrix=np.asarray([[1.0], [1.0]]),
        candidate_tokens=("candidate:first", "candidate:second"),
    )
    solver.record_evaluation_event(
        batch,
        OptimizationFeedbackBatch.from_feedback((first_feedback, second_feedback)),
        (
            CandidateProvenance(candidate_token="candidate:first"),
            CandidateProvenance(candidate_token="candidate:second"),
        ),
    )
    solver.set_incumbent(
        IncumbentState(
            candidate=[1.0],
            objectives=[1.0],
            constraint_violation=0.0,
            score=1.0,
            policy_id=str(solver.incumbent_scalarizer_id),
            policy_context=dict(solver.incumbent_scalarizer_context),
            candidate_token="candidate:second",
            metadata={"candidate.semantic_metadata": {"architecture": "second"}},
        )
    )

    solver._synchronize_ml_projection(record_history=True)

    assert solver.best_model == "second-model"
    assert solver.best_feedback.metrics["source"] == "second"
    assert solver.best_state.metadata["architecture"] == "second"
    assert solver.history[-1]["metrics"] == {"source": "second"}
    assert solver.history[-1]["metrics_candidate_token"] == "candidate:second"


def test_learning_uses_declared_ports_instead_of_replacing_control_methods():
    solver = _solver()
    for method in ("step", "build_context", "setup", "prepare_fresh_run", "teardown",
                   "checkpoint_components", "should_execute_step", "init_population"):
        assert method not in type(solver).__dict__
    assert solver.build_context()["run_name"] == solver.run_name
    assert solver.build_context()["trajectory_id"] == (solver.run_session.run_id or solver.run_name)
    assert "model_representation" in solver.checkpoint_components()


def test_learning_teardown_attempts_all_participants_and_preserves_primary(
    monkeypatch,
) -> None:
    from blackbase.project import collect_failure_evidence

    solver = _solver()
    events: list[str] = []

    def fail_solver_teardown(_self) -> None:
        events.append("solver")
        raise ValueError("solver teardown failed")

    def fail_provider_release() -> None:
        events.append("provider")
        raise RuntimeError("provider release failed")

    def fail_backend_close() -> None:
        events.append("backend")
        raise OSError("backend close failed")

    monkeypatch.setattr(solver.adapter, "teardown", fail_solver_teardown)
    solver._release_provider_trajectory = fail_provider_release
    solver.compute_backend_session.close = fail_backend_close

    with pytest.raises(ValueError, match="solver teardown failed") as captured:
        solver.teardown()

    assert events == ["solver", "provider", "backend"]
    evidence = collect_failure_evidence(captured.value)["solver_teardown"]
    assert [item["phase"] for item in evidence] == [
        "adapter",
        "ml_projection.provider_trajectory",
        "ml_projection.compute_backend",
    ]


def test_missing_canonical_feedback_clears_stale_projection() -> None:
    solver = _solver()
    solver.best_feedback = Feedback(objectives=[9.0], metrics={"source": "old"})
    solver._best_feedback_binding = {"candidate_token": "candidate:old"}
    solver.set_incumbent(
        IncumbentState(
            candidate=[2.0],
            objectives=[1.0],
            constraint_violation=0.0,
            score=1.0,
            policy_id=str(solver.incumbent_scalarizer_id),
            policy_context=dict(solver.incumbent_scalarizer_context),
            candidate_token="candidate:new",
        )
    )

    with pytest.raises(RuntimeError, match="no token-matched canonical feedback"):
        solver._synchronize_ml_projection(record_history=False)
    assert solver.best_feedback is None


def test_large_trainer_result_fields_are_published_before_case_serialization() -> None:
    solver = _solver(size=512, inline_limit=1024)
    runtime = _CaseRuntime()
    solver.set_case_runtime(runtime)

    result = solver.fit(max_steps=20)
    payload = result.as_dict()

    assert result.best_model_ref is not None
    assert result.best_state_ref is not None
    assert result.history_ref is not None
    assert result.report_ref is not None
    assert payload["best_model"] is None
    assert payload["best_state"] is None
    assert payload["history"] == []
    assert payload["report"] == {}
    assert payload["best_state_ref"]["uri"].endswith("/trainer_best_state")
    assert payload["history_ref"]["uri"].endswith("/trainer_history")
    assert payload["report_ref"]["uri"].endswith("/trainer_report")
    assert set(runtime.finalization_artifact_refs).issuperset(
        {"best_model", "trainer_best_state", "trainer_history", "trainer_report"}
    )


def test_final_artifacts_are_published_only_after_successful_teardown() -> None:
    solver = _solver(size=512, inline_limit=1024)
    runtime = _CaseRuntime()
    events: list[str] = []
    original_publish = runtime._materialize_artifact
    original_teardown = solver.teardown

    def publish_artifact(name, value, **kwargs):
        events.append(f"publish:{name}")
        assert "teardown" in events
        return original_publish(name, value, **kwargs)

    def teardown() -> None:
        original_teardown()
        events.append("teardown")

    runtime._materialize_artifact = publish_artifact  # type: ignore[method-assign]
    solver.teardown = teardown  # type: ignore[method-assign]
    solver.set_case_runtime(runtime)

    solver.fit(max_steps=1)

    assert events[0] == "teardown"
    assert any(item.startswith("publish:") for item in events[1:])


def test_teardown_failure_publishes_no_success_artifacts() -> None:
    solver = _solver(size=512, inline_limit=1024)
    runtime = _CaseRuntime()
    solver.set_case_runtime(runtime)

    def failing_teardown() -> None:
        raise RuntimeError("provider close failed")

    solver.teardown = failing_teardown  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="provider close failed"):
        solver.fit(max_steps=1)

    assert runtime.artifact_refs == {}
    assert solver.result_artifact_refs == {}
    assert solver._finalized_trainer_result is None


def test_partial_artifact_staging_failure_aborts_all_formal_refs() -> None:
    solver = _solver(size=512, inline_limit=1024)
    runtime = _CaseRuntime()
    original_materialize = runtime._materialize_artifact
    calls = 0

    def fail_second_artifact(name, value, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("artifact store failed")
        return original_materialize(name, value, **kwargs)

    runtime._materialize_artifact = fail_second_artifact  # type: ignore[method-assign]
    solver.set_case_runtime(runtime)

    with pytest.raises(RuntimeError, match="artifact store failed"):
        solver.fit(max_steps=1)

    assert calls == 2
    assert runtime.artifact_refs == {}
    assert solver.result_artifact_refs == {}
    assert solver._finalized_trainer_result is None


def test_strict_finalization_prepare_failure_aborts_staged_artifacts() -> None:
    solver = _solver(size=512, inline_limit=1024, plugin_strict=True)
    runtime = _CaseRuntime()

    class _FailingFinalizationPrepareHook(Plugin):
        participation = ParticipationContract(vetoes=("finalization",))
        def __init__(self) -> None:
            super().__init__(name="failing_finalization_prepare_hook")

        def on_solver_finalization_prepare(self, result) -> None:
            assert result["trainer_result_ready"] is False
            assert runtime.artifact_refs == {}
            raise RuntimeError("finalization prepare rejected publication")

    solver.add_plugin(_FailingFinalizationPrepareHook())
    solver.set_case_runtime(runtime)

    with pytest.raises(RuntimeError, match="finalization prepare rejected publication"):
        solver.fit(max_steps=1)

    assert runtime.artifact_refs == {}
    assert solver.result_artifact_refs == {}
    assert solver._finalized_trainer_result is None


def test_finalized_hook_observes_committed_trainer_result_and_artifacts() -> None:
    solver = _solver(size=512, inline_limit=1024, plugin_strict=True)
    runtime = _CaseRuntime()
    events: list[str] = []

    class _FinalizationObserver(Plugin):
        participation = ParticipationContract(vetoes=("finalization",))
        def __init__(self) -> None:
            super().__init__(name="finalization_observer")

        def on_solver_finalization_prepare(self, result) -> None:
            assert result["trainer_result_ready"] is False
            assert runtime.artifact_refs == {}
            events.append("prepare")

        def on_solver_finalized(self, result) -> None:
            assert result["trainer_result_ready"] is True
            assert runtime.finalization_artifact_refs
            assert solver._finalized_trainer_result is not None
            events.append("finalized")

    solver.add_plugin(_FinalizationObserver())
    solver.set_case_runtime(runtime)

    result = solver.fit(max_steps=1)

    assert result is solver._finalized_trainer_result
    assert events == ["prepare"]
    assert runtime.seal_finalization() == ()
    assert events == ["prepare", "finalized"]


def test_finalized_observer_failure_is_post_seal_diagnostic() -> None:
    solver = _solver(size=512, inline_limit=1024, plugin_strict=True)
    runtime = _CaseRuntime()

    class _FailingFinalizedObserver(Plugin):
        def __init__(self) -> None:
            super().__init__(name="failing_finalized_observer")

        def on_solver_finalized(self, result) -> None:
            assert result["trainer_result_ready"] is True
            assert runtime.finalization_artifact_refs
            raise RuntimeError("finalized observer failed")

    solver.add_plugin(_FailingFinalizedObserver())
    solver.set_case_runtime(runtime)

    result = solver.fit(max_steps=1)
    failures = runtime.seal_finalization()

    assert result is solver._finalized_trainer_result
    assert set(runtime.artifact_refs) == set(runtime.finalization_artifact_refs)
    assert solver.result_artifact_refs
    assert len(failures) == 1
    assert "finalized observer failed" in str(failures[0][1])


def test_direct_ml_final_artifact_publication_is_rejected() -> None:
    solver = _solver()

    with pytest.raises(RuntimeError, match="unified Solver lifecycle"):
        solver.finalize_after_teardown({})
    with pytest.raises(RuntimeError, match="unified Solver finalization transaction"):
        solver.publish_best_model_artifact()

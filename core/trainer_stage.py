"""
Stage orchestration for the Trainer control plane.

Serial/parallel trainer chains with artifact flow between stages.
Mirrors nsgablack's solver_stage.py for the Trainer lifecycle.
"""

from __future__ import annotations

import inspect
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from blackbase.context import unwrap_snapshot_payload

from .trainer import BlankTrainer
from .types import TrainerResult


# ── helpers ────────────────────────────────────────────────────────────
def _ref_to_dict(ref: ArtifactRef) -> Dict[str, Any]:
    d: Dict[str, Any] = {"key": str(ref.key), "uri": str(ref.uri), "kind": str(ref.kind)}
    if ref.backend:
        d["backend"] = str(ref.backend)
    if ref.schema:
        d["schema"] = str(ref.schema)
    if ref.meta:
        d["meta"] = dict(ref.meta)
    if ref.inline_value is not None:
        d["has_inline"] = True
    return d


def _is_small_payload(obj: Any) -> bool:
    try:
        return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) <= 1024
    except Exception:
        return False


# ────────────────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArtifactRef:
    """Typed reference to an artifact produced by one stage, consumed by another.

    Large artifacts live in snapshot stores; small payloads may be inlined.
    """

    key: str
    uri: str
    kind: str = "snapshot"  # "snapshot" | "inline"
    backend: str = "snapshot_store"
    schema: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    inline_value: Optional[Any] = None

    def resolve(self, snapshot_store: Optional[Any] = None) -> Any:
        if self.inline_value is not None:
            return self.inline_value
        if snapshot_store is None:
            raise RuntimeError(f"Cannot resolve artifact '{self.key}': no snapshot_store")
        if callable(getattr(snapshot_store, "read", None)):
            record = snapshot_store.read(self.uri)
        elif hasattr(snapshot_store, "__getitem__"):
            record = snapshot_store[self.uri]
        else:
            raise RuntimeError(f"Snapshot store cannot resolve '{self.uri}'")
        if record is None:
            raise KeyError(f"Artifact '{self.key}' not found at '{self.uri}'")
        raw_payload = record.payload if hasattr(record, "payload") else record
        payload = unwrap_snapshot_payload(raw_payload)
        # Read snapshots produced by older BlankTrainer.write_snapshot().
        if isinstance(payload, Mapping) and set(payload) == {str(self.uri)}:
            return payload[str(self.uri)]
        return payload


@dataclass(frozen=True)
class CompletionPolicy:
    """Declares when a stage is considered complete."""

    max_steps: Optional[int] = None
    max_seconds: Optional[float] = None
    custom_check: Optional[Callable[[Dict[str, Any]], bool]] = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and int(self.max_steps) < 0:
            raise ValueError("CompletionPolicy.max_steps must be >= 0")
        if self.max_seconds is not None and float(self.max_seconds) < 0.0:
            raise ValueError("CompletionPolicy.max_seconds must be >= 0")

    def is_complete(self, *, step: int, elapsed: float, ctx: Dict[str, Any]) -> bool:
        if self.max_steps is not None and step >= self.max_steps:
            return True
        if self.max_seconds is not None and elapsed >= self.max_seconds:
            return True
        if callable(self.custom_check):
            if self.custom_check({"step": step, "elapsed": elapsed, **ctx}):
                return True
        return False


@dataclass(frozen=True)
class StageSpec:
    """One stage in an orchestrated training pipeline."""

    name: str
    factory: Callable[[], Any]  # () -> Trainer
    completion: CompletionPolicy = field(default_factory=CompletionPolicy)
    input_artifacts: Dict[str, str] = field(default_factory=dict)
    # ^ trainer_key → registry_key
    output_artifacts: List[str] = field(default_factory=list)
    # ^ registry keys produced by this stage
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────
# SerialTrainer
# ────────────────────────────────────────────────────────────────────────


class SerialTrainer(BlankTrainer):
    """Chain multiple trainers in sequence with artifact flow between stages.

    Presents as a normal ``BlankTrainer`` so it can be used anywhere a
    Trainer is expected — including as an inner trainer inside an
    nsgablack ``SerialStageSolver``.
    """

    def __init__(
        self,
        stages: List[StageSpec],
        *,
        run_name: str = "serial_trainer",
        constraint_penalty: float = 1e6,
        resource_context: Optional[Mapping[str, Any]] = None,
        compute_backend: Optional[Any] = None,
        output_stage: Optional[str] = None,
        result_aggregator: Optional[
            Callable[[Sequence[TrainerResult]], TrainerResult]
        ] = None,
    ) -> None:
        super().__init__(
            run_name=run_name,
            constraint_penalty=constraint_penalty,
            resource_context=resource_context,
            compute_backend=compute_backend,
        )
        self._stages: List[StageSpec] = [s for s in stages if s.enabled]
        if not self._stages:
            raise ValueError("SerialTrainer requires at least one enabled stage")
        stage_names = [str(stage.name) for stage in self._stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("SerialTrainer stage names must be unique")
        self.output_stage = str(output_stage or stage_names[-1])
        if self.output_stage not in set(stage_names):
            raise ValueError(
                f"SerialTrainer output_stage '{self.output_stage}' is not an enabled stage"
            )
        self.result_aggregator = result_aggregator
        self._artifact_registry: Dict[str, ArtifactRef] = {}
        self._stage_results: List[Dict[str, Any]] = []

    # -- public read-only -------------------------------------------------

    @property
    def artifact_registry(self) -> Dict[str, ArtifactRef]:
        return dict(self._artifact_registry)

    @property
    def stage_results(self) -> List[Dict[str, Any]]:
        return list(self._stage_results)

    def get_artifact(self, key: str) -> Optional[ArtifactRef]:
        return self._artifact_registry.get(key)

    # -- lifecycle --------------------------------------------------------

    def setup(self) -> None:
        pass

    def step(self, context: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        return {}

    def teardown(self) -> None:
        pass

    def fit(self, max_steps: int = 100) -> TrainerResult:
        _ = max_steps  # per-stage policies override this
        total = len(self._stages)
        self._artifact_registry.clear()
        self._stage_results.clear()
        self.history.clear()
        child_results: list[TrainerResult] = []
        output_result: Optional[TrainerResult] = None
        teardown_called = False
        active_trainer: Any = None
        active_stage: Optional[StageSpec] = None

        try:
            self.setup()
            self.plugin_manager.on_solver_init(self)
            for idx, stage in enumerate(self._stages):
                trainer = stage.factory()
                active_trainer = trainer
                active_stage = stage

                # The SerialTrainer is the parent Case runtime. Child stages consume
                # a derived grant and the same storage backends; they do not mint a
                # private lease or create an unreachable artifact namespace.
                child_resource = self.resource_context.derive_child(
                    scope="training_stage",
                    namespace_suffix=f"stage.{idx}.{stage.name}",
                    threads=int(self.resource_context.threads),
                    metadata={"stage_name": stage.name, "stage_index": idx},
                )
                set_resource_context = getattr(trainer, "set_resource_context", None)
                if callable(set_resource_context):
                    set_resource_context(child_resource)
                else:
                    setattr(trainer, "resource_context", child_resource)
                set_context_store = getattr(trainer, "set_context_store", None)
                if callable(set_context_store):
                    set_context_store(self.context_store)
                else:
                    setattr(trainer, "context_store", self.context_store)
                set_snapshot_store = getattr(trainer, "set_snapshot_store", None)
                if callable(set_snapshot_store):
                    set_snapshot_store(self.snapshot_store)
                else:
                    setattr(trainer, "snapshot_store", self.snapshot_store)

                # inject stage metadata
                self._inject_stage_meta(trainer, stage, idx, total)

                # inject input artifacts
                self._inject_artifacts(trainer, stage)

                # record stage start
                start_time = time.time()

                # run
                max_s = stage.completion.max_steps
                step_count = 0

                # Prefer fit() (standard Trainer API), fall back to run()/step()
                if callable(getattr(trainer, "fit", None)):
                    self._install_cooperative_completion_policy(trainer, stage)
                    result = self._call_with_max_steps(trainer.fit, max_s, stage.name)
                elif callable(getattr(trainer, "run", None)):
                    self._install_cooperative_completion_policy(trainer, stage)
                    set_max_steps = getattr(trainer, "set_max_steps", None)
                    if max_s is not None and callable(set_max_steps):
                        set_max_steps(max_s)
                        result = trainer.run()
                    else:
                        result = self._call_with_max_steps(trainer.run, max_s, stage.name)
                elif callable(getattr(trainer, "step", None)):
                    if (
                        stage.completion.max_steps is None
                        and stage.completion.max_seconds is None
                        and stage.completion.custom_check is None
                    ):
                        raise ValueError(
                            f"Stage '{stage.name}' uses step() but has no completion condition"
                        )
                    trainer.setup() if callable(getattr(trainer, "setup", None)) else None
                    try:
                        while not stage.completion.is_complete(
                            step=step_count,
                            elapsed=time.time() - start_time,
                            ctx={"trainer": trainer, "stage": stage.name},
                        ):
                            trainer.step()
                            step_count += 1
                    finally:
                        trainer.teardown() if callable(getattr(trainer, "teardown", None)) else None
                    result = {"steps": step_count}
                else:
                    raise RuntimeError(f"Stage '{stage.name}' trainer has no fit(), run(), or step()")

                elapsed = time.time() - start_time
                trainer_result = self._coerce_stage_result(trainer, result)
                child_results.append(trainer_result)
                if str(stage.name) == self.output_stage:
                    output_result = trainer_result

                # extract output artifacts
                self._extract_artifacts(trainer, stage, idx)
                self._set_stage_status(trainer, "completed")

                self._stage_results.append({
                    "stage_name": stage.name,
                    "stage_index": idx,
                    "elapsed_sec": elapsed,
                    "status": "completed",
                    "result": self._stage_result_summary(trainer_result),
                    "produced_artifacts": [
                        k for k in stage.output_artifacts if k in self._artifact_registry
                    ],
                })
                self.history = list(self._stage_results)
                self.step_index = int(idx)
                active_trainer = None
                active_stage = None

            if self.result_aggregator is not None:
                output_result = self.result_aggregator(tuple(child_results))
                if not isinstance(output_result, TrainerResult):
                    raise TypeError("SerialTrainer result_aggregator must return TrainerResult")
            if output_result is None:
                raise RuntimeError(
                    f"SerialTrainer output stage '{self.output_stage}' produced no result"
                )
            self._adopt_result(output_result)
            teardown_called = True
            self.teardown()
            report = self.build_report()
            result = TrainerResult(
                best_state=self.best_state,
                best_model=self.best_model,
                best_objectives=output_result.best_objectives,
                best_feedback=self.best_feedback,
                history=tuple(self._stage_results),
                population=output_result.population,
                report=report,
                metadata={
                    **dict(output_result.metadata),
                    "output_stage": self.output_stage,
                    "result_policy": (
                        "aggregator"
                        if self.result_aggregator is not None
                        else "selected_stage"
                    ),
                },
            )
            self.plugin_manager.on_solver_finish(
                {"report": report, "context": self.build_context(), "result": result}
            )
            return result
        except BaseException as exc:
            if active_trainer is not None:
                self._set_stage_status(active_trainer, "failed")
            self.plugin_manager.on_error(
                exc,
                self.build_context(
                    {
                        "error": type(exc).__name__,
                        "stage_name": None if active_stage is None else active_stage.name,
                        "stage_status": "failed",
                    }
                ),
            )
            raise
        finally:
            if not teardown_called:
                self.teardown()

    def run(self, max_steps: int = 100) -> TrainerResult:
        """Execute the composite lifecycle through the overridden ``fit`` path."""

        return self.fit(max_steps=max_steps)

    @staticmethod
    def _coerce_stage_result(trainer: Any, result: Any) -> TrainerResult:
        if isinstance(result, TrainerResult):
            return result
        return TrainerResult(
            best_state=getattr(trainer, "best_state", None),
            best_model=getattr(trainer, "best_model", None),
            best_feedback=getattr(trainer, "best_feedback", None),
            history=tuple(getattr(trainer, "history", ()) or ()),
            report=(dict(result) if isinstance(result, Mapping) else {"raw": result}),
        )

    @staticmethod
    def _stage_result_summary(result: TrainerResult) -> Dict[str, Any]:
        return {
            "best_state": None
            if result.best_state is None
            else getattr(result.best_state, "as_dict", lambda: result.best_state)(),
            "best_objectives": None
            if result.best_objectives is None
            else result.best_objectives.tolist(),
            "has_best_model": result.best_model is not None,
            "has_best_feedback": result.best_feedback is not None,
            "report": dict(result.report),
            "metadata": dict(result.metadata),
        }

    def _adopt_result(self, result: TrainerResult) -> None:
        self.best_state = result.best_state
        self.best_model = result.best_model
        self.best_feedback = result.best_feedback
        if self.best_feedback is not None:
            self.best_score = self.best_feedback.scalar_score(
                constraint_penalty=self.constraint_penalty
            )
        elif result.best_objectives is not None and result.best_objectives.size:
            self.best_score = float(result.best_objectives.mean())
        else:
            self.best_score = None

    # -- artifact flow ----------------------------------------------------

    def _inject_artifacts(self, trainer: Any, stage: StageSpec) -> None:
        if not stage.input_artifacts:
            return
        for trainer_key, registry_key in stage.input_artifacts.items():
            ref = self._artifact_registry.get(registry_key)
            if ref is None:
                raise KeyError(
                    f"Stage '{stage.name}' requires missing artifact '{registry_key}'"
                )
            resolved = ref.resolve(getattr(trainer, "snapshot_store", self.snapshot_store))
            # try setter, then attribute
            setter = getattr(trainer, f"set_{trainer_key}", None)
            if callable(setter):
                setter(resolved)
                continue
            setattr(trainer, trainer_key, resolved)

    def _extract_artifacts(self, trainer: Any, stage: StageSpec, stage_index: int) -> None:
        if not stage.output_artifacts:
            return

        for artifact_key in stage.output_artifacts:
            ref: Optional[ArtifactRef] = None
            raw_artifact: Any = None

            # route 1: getter
            getter = getattr(trainer, f"get_{artifact_key}_artifact", None)
            if callable(getter):
                raw_artifact = getter()
                if isinstance(raw_artifact, ArtifactRef):
                    ref = raw_artifact

            # route 2: trainer attribute
            if ref is None:
                attr = raw_artifact if raw_artifact is not None else getattr(trainer, artifact_key, None)
                if attr is not None:
                    if isinstance(attr, ArtifactRef):
                        ref = attr
                    elif _is_small_payload(attr):
                        ref = ArtifactRef(
                            key=artifact_key, uri="inline", kind="inline", inline_value=attr
                        )
                    else:
                        snap_key = f"stage_{stage_index}.{artifact_key}"
                        writer = getattr(trainer, "write_snapshot", None)
                        if not callable(writer):
                            writer = self.write_snapshot
                        written_key = writer(attr, key=snap_key)
                        ref = ArtifactRef(
                            key=artifact_key,
                            uri=str(written_key or snap_key),
                            kind="snapshot",
                            meta={"stage_index": stage_index},
                        )

            if ref is None:
                raise RuntimeError(
                    f"Stage '{stage.name}' declared output artifact '{artifact_key}' "
                    "but produced no value"
                )
            self._artifact_registry[artifact_key] = ref

    # -- stage metadata ---------------------------------------------------

    def _inject_stage_meta(self, trainer: Any, stage: StageSpec, idx: int, total: int) -> None:
        store = getattr(trainer, "context_store", None)
        if store is not None and callable(getattr(store, "__setitem__", None)):
            store["stage_index"] = idx
            store["stage_name"] = stage.name
            store["stage_total"] = total
            store["stage_status"] = "running"

    @staticmethod
    def _set_stage_status(trainer: Any, status: str) -> None:
        store = getattr(trainer, "context_store", None)
        if store is not None and callable(getattr(store, "__setitem__", None)):
            store["stage_status"] = str(status)

    @staticmethod
    def _call_with_max_steps(callable_obj: Callable[..., Any], max_steps: Optional[int], stage_name: str) -> Any:
        if max_steps is None:
            return callable_obj()
        parameters = inspect.signature(callable_obj).parameters
        accepts_keyword = "max_steps" in parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if not accepts_keyword:
            raise ValueError(
                f"Stage '{stage_name}' declares max_steps but its entrypoint "
                "does not accept max_steps or provide set_max_steps()"
            )
        return callable_obj(max_steps=int(max_steps))

    @staticmethod
    def _install_cooperative_completion_policy(trainer: Any, stage: StageSpec) -> None:
        policy = stage.completion
        if policy.max_seconds is None and policy.custom_check is None:
            return
        setter = getattr(trainer, "set_completion_policy", None)
        if not callable(setter):
            raise ValueError(
                f"Stage '{stage.name}' declares max_seconds/custom_check but its "
                "entrypoint does not support cooperative completion policies"
            )
        setter(policy)

    # -- result collection ------------------------------------------------

    def build_report(self) -> Dict[str, Any]:
        report = super().build_report()
        report["stage_count"] = len(self._stage_results)
        report["stages"] = [
            {
                "name": r["stage_name"],
                "index": r["stage_index"],
                "elapsed_sec": r["elapsed_sec"],
                "status": r["status"],
                "artifacts": r["produced_artifacts"],
            }
            for r in self._stage_results
        ]
        report["artifact_registry"] = {
            k: _ref_to_dict(v) for k, v in self._artifact_registry.items()
        }
        report["output_stage"] = self.output_stage
        report["result_policy"] = (
            "aggregator" if self.result_aggregator is not None else "selected_stage"
        )
        return report

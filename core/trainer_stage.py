"""
Stage orchestration for the Trainer control plane.

Serial/parallel trainer chains with artifact flow between stages.
Mirrors nsgablack's solver_stage.py for the Trainer lifecycle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

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
        return len(repr(obj).encode("utf-8")) <= 1024
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
        if hasattr(record, "payload"):
            return record.payload
        return record


@dataclass(frozen=True)
class CompletionPolicy:
    """Declares when a stage is considered complete."""

    max_steps: Optional[int] = None
    max_seconds: Optional[float] = None
    custom_check: Optional[Callable[[Dict[str, Any]], bool]] = None

    def is_complete(self, *, step: int, elapsed: float, ctx: Dict[str, Any]) -> bool:
        if self.max_steps is not None and step >= self.max_steps:
            return True
        if self.max_seconds is not None and elapsed >= self.max_seconds:
            return True
        if callable(self.custom_check):
            try:
                if self.custom_check({"step": step, "elapsed": elapsed, **ctx}):
                    return True
            except Exception:
                pass
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

        for idx, stage in enumerate(self._stages):
            trainer = stage.factory()

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
                result = trainer.fit(max_steps=max_s) if max_s is not None else trainer.fit()
            elif callable(getattr(trainer, "run", None)):
                if max_s is not None and callable(getattr(trainer, "set_max_steps", None)):
                    trainer.set_max_steps(max_s)
                result = trainer.run()
            elif callable(getattr(trainer, "step", None)):
                trainer.setup() if callable(getattr(trainer, "setup", None)) else None
                while not stage.completion.is_complete(
                    step=step_count,
                    elapsed=time.time() - start_time,
                    ctx={"trainer": trainer, "stage": stage.name},
                ):
                    trainer.step()
                    step_count += 1
                trainer.teardown() if callable(getattr(trainer, "teardown", None)) else None
                result = {"steps": step_count}
            else:
                raise RuntimeError(f"Stage '{stage.name}' trainer has no fit(), run(), or step()")

            elapsed = time.time() - start_time

            # extract output artifacts
            self._extract_artifacts(trainer, stage, idx)

            self._stage_results.append({
                "stage_name": stage.name,
                "stage_index": idx,
                "elapsed_sec": elapsed,
                "result": result if isinstance(result, dict) else {"raw": result},
                "produced_artifacts": [
                    k for k in stage.output_artifacts if k in self._artifact_registry
                ],
            })

        return TrainerResult(
            best_state=self.best_state,
            best_model=self.best_model,
            best_feedback=self.best_feedback,
            history=tuple(self._stage_results),
            report=self.build_report(),
        )

    # -- artifact flow ----------------------------------------------------

    def _inject_artifacts(self, trainer: Any, stage: StageSpec) -> None:
        if not stage.input_artifacts:
            return
        for trainer_key, registry_key in stage.input_artifacts.items():
            ref = self._artifact_registry.get(registry_key)
            if ref is None:
                continue
            # try setter, then attribute, then context
            setter = getattr(trainer, f"set_{trainer_key}", None)
            if callable(setter):
                try:
                    setter(ref.resolve(getattr(trainer, "snapshot_store", None)))
                    continue
                except Exception:
                    pass
            # always attempt attribute injection
            setattr(trainer, trainer_key, ref)

    def _extract_artifacts(self, trainer: Any, stage: StageSpec, stage_index: int) -> None:
        if not stage.output_artifacts:
            return

        for artifact_key in stage.output_artifacts:
            ref: Optional[ArtifactRef] = None

            # route 1: getter
            getter = getattr(trainer, f"get_{artifact_key}_artifact", None)
            if callable(getter):
                try:
                    raw = getter()
                    if isinstance(raw, ArtifactRef):
                        ref = raw
                except Exception:
                    pass

            # route 2: trainer attribute
            if ref is None:
                attr = getattr(trainer, artifact_key, None)
                if attr is not None:
                    if isinstance(attr, ArtifactRef):
                        ref = attr
                    elif _is_small_payload(attr):
                        ref = ArtifactRef(
                            key=artifact_key, uri="inline", kind="inline", inline_value=attr
                        )
                    else:
                        snap_key = f"stage_{stage_index}.{artifact_key}"
                        if callable(getattr(trainer, "write_snapshot", None)):
                            trainer.write_snapshot(attr, key=snap_key)
                        ref = ArtifactRef(
                            key=artifact_key, uri=snap_key, kind="snapshot",
                            meta={"stage_index": stage_index},
                        )

            if ref is not None:
                self._artifact_registry[artifact_key] = ref

    # -- stage metadata ---------------------------------------------------

    def _inject_stage_meta(self, trainer: Any, stage: StageSpec, idx: int, total: int) -> None:
        store = getattr(trainer, "context_store", None)
        if store is not None and callable(getattr(store, "__setitem__", None)):
            store["stage_index"] = idx
            store["stage_name"] = stage.name
            store["stage_total"] = total
            store["stage_status"] = "running"

    # -- result collection ------------------------------------------------

    def build_report(self) -> Dict[str, Any]:
        report = super().build_report()
        report["stage_count"] = len(self._stage_results)
        report["stages"] = [
            {
                "name": r["stage_name"],
                "index": r["stage_index"],
                "elapsed_sec": r["elapsed_sec"],
                "artifacts": r["produced_artifacts"],
            }
            for r in self._stage_results
        ]
        report["artifact_registry"] = {
            k: _ref_to_dict(v) for k, v in self._artifact_registry.items()
        }
        return report

from __future__ import annotations

from dataclasses import dataclass, field
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from mlblack.core.capability import Capability
from blackbase.contracts import ComponentContract


@dataclass(frozen=True)
class ExperimentRecord:
    record_id: str
    run_name: str
    event: str
    step: int | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "run_name": self.run_name,
            "event": self.event,
            "step": self.step,
            "payload": dict(self.payload),
        }


class InMemoryExperimentStore:
    def __init__(self) -> None:
        self.records: list[ExperimentRecord] = []

    def write(self, record: ExperimentRecord) -> None:
        self.records.append(record)

    def list(self, *, run_name: str | None = None) -> tuple[ExperimentRecord, ...]:
        if run_name is None:
            return tuple(self.records)
        return tuple(record for record in self.records if record.run_name == str(run_name))


class SQLiteExperimentStore:
    def __init__(self, path: str | Path, *, timeout: float = 30.0) -> None:
        self.path = str(path)
        self.timeout = float(timeout)
        self._init_schema()

    def write(self, record: ExperimentRecord) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO experiment_records (record_id, run_name, event, step, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.run_name,
                    record.event,
                    record.step,
                    json.dumps(dict(record.payload), ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list(self, *, run_name: str | None = None) -> tuple[ExperimentRecord, ...]:
        conn = self._connect()
        try:
            if run_name is None:
                rows = conn.execute(
                    "SELECT record_id, run_name, event, step, payload_json FROM experiment_records ORDER BY rowid ASC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT record_id, run_name, event, step, payload_json FROM experiment_records WHERE run_name = ? ORDER BY rowid ASC",
                    (str(run_name),),
                ).fetchall()
        finally:
            conn.close()
        return tuple(
            ExperimentRecord(
                record_id=str(row["record_id"]),
                run_name=str(row["run_name"]),
                event=str(row["event"]),
                step=row["step"],
                payload=_json_loads(str(row["payload_json"])),
            )
            for row in rows
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_records (
                    record_id TEXT PRIMARY KEY,
                    run_name TEXT NOT NULL,
                    event TEXT NOT NULL,
                    step INTEGER,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiment_records_run ON experiment_records(run_name)")
            conn.commit()
        finally:
            conn.close()


@dataclass(frozen=True)
class ExperimentTrackerConfig:
    record_steps: bool = True
    record_evaluations: bool = False


class ExperimentTrackerCapability(Capability):
    name = "experiment_tracker"
    context_requires = ()
    context_optional = ('trainer.context', 'trainer.report')
    context_provides = ('experiment.records',)
    context_mutates = ('trainer.context',)
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = 'provides experiment.records; mutates trainer.context.'
    contract = ComponentContract(
        name=name,
        optional=("trainer.context", "trainer.report"),
        provides=("experiment.records",),
        mutates=("trainer.context",),
        supports_resume=True,
        metadata={"capability": "experiment_tracker"},
    )

    def __init__(
        self,
        store: InMemoryExperimentStore | SQLiteExperimentStore | None = None,
        config: ExperimentTrackerConfig | None = None,
    ) -> None:
        self.store = store or InMemoryExperimentStore()
        self.config = config or ExperimentTrackerConfig()

    def on_fit_start(self, trainer: Any, context: Mapping[str, Any]) -> None:
        self._write(trainer, "fit_start", context)

    def on_step_end(self, trainer: Any, context: Mapping[str, Any], row: Mapping[str, Any]) -> None:
        if self.config.record_steps:
            self._write(trainer, "step_end", {**dict(context), "row": dict(row)}, step=int(row.get("step", 0)))

    def on_evaluate_end(self, trainer: Any, candidate: Any, feedback: Any, context: Mapping[str, Any]) -> None:
        if not self.config.record_evaluations:
            return
        metrics = dict(getattr(feedback, "metrics", {}) or {})
        self._write(trainer, "evaluate_end", {"metrics": metrics, "context": dict(context)}, step=int(context.get("step", 0)))

    def on_fit_end(self, trainer: Any, context: Mapping[str, Any], report: Mapping[str, Any]) -> None:
        self._write(trainer, "fit_end", {"context": dict(context), "report": dict(report)})
        trainer.context_store["experiment.record_count"] = len(self.store.list(run_name=str(getattr(trainer, "run_name", ""))))

    def on_error(self, trainer: Any, error: BaseException, context: Mapping[str, Any]) -> None:
        self._write(trainer, "error", {"error": repr(error), "context": dict(context)}, step=int(context.get("step", 0)))

    def _write(self, trainer: Any, event: str, payload: Mapping[str, Any], step: int | None = None) -> None:
        record = ExperimentRecord(
            record_id=f"rec_{uuid4().hex[:16]}",
            run_name=str(getattr(trainer, "run_name", "run")),
            event=str(event),
            step=step,
            payload=_safe_payload(payload),
        )
        self.store.write(record)


def _safe_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in dict(payload).items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _json_loads(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


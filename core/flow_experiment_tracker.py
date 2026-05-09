from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence

from core.experiment_db import (
    decode_row,
    ensure_table_columns,
    first_column_texts,
    open_experiment_db,
    resolve_experiment_db_target,
    table_columns,
    table_count,
    table_exists,
)
from core.state.context_keys import RUN_STAGE
from experiment.contracts import (
    ArtifactRecord,
    AssemblyRecord,
    RunRecord,
    SurfaceRecord,
    make_artifact_record,
    make_assembly_record,
    make_run_record,
    make_surface_record,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    try:
        import numpy as np  # local import to avoid hard dependency at import time

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False)


def _json_loads(value: Any) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


class ExperimentTrackerCapability:
    """Lifecycle capability that records experiment surfaces into the resolved DB target."""

    def __init__(
        self,
        *,
        name: str = "experiment_tracker",
        db_path: str = "runs/experiments.sqlite3",
        namespace: str = "default",
        tag: str | None = None,
        report_key: str = "experiment_tracker",
        max_payload_chars: int = 25000,
        io_mode: str = "batched",
        commit_interval: int = 12,
        priority: int = 0,
        enabled: bool = True,
        is_algorithmic: bool = False,
        config: Dict[str, Any] | None = None,
        context_requires: Sequence[str] = ("run_name",),
        context_provides: Sequence[str] = ("experiment_tracker",),
        context_mutates: Sequence[str] = ("report",),
        context_cache: Sequence[str] = tuple(),
        context_notes: str | None = "Persists run events and metrics to the resolved experiment DB target for experiment visualization.",
    ) -> None:
        self.name = str(name)
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.is_algorithmic = bool(is_algorithmic)
        self.config = dict(config or {})
        self.context_requires = tuple(str(x) for x in context_requires)
        self.context_provides = tuple(str(x) for x in context_provides)
        self.context_mutates = tuple(str(x) for x in context_mutates)
        self.context_cache = tuple(str(x) for x in context_cache)
        self.context_notes = None if context_notes is None else str(context_notes)

        self.db_path = resolve_experiment_db_target(str(db_path).strip())
        if not self.db_path:
            raise ValueError("experiment_tracker requires non-empty db_path")
        self.namespace = str(namespace or "default").strip() or "default"
        self.tag = None if tag is None else str(tag)
        self.report_key = str(report_key or "experiment_tracker")
        self.max_payload_chars = max(1000, int(max_payload_chars))
        self.io_mode = self._normalize_io_mode(io_mode)
        try:
            interval = int(commit_interval)
        except Exception as exc:
            raise ValueError("experiment_tracker commit_interval must be int") from exc
        if self.io_mode == "safe":
            self.commit_interval = max(1, interval)
        elif self.io_mode == "legacy":
            self.commit_interval = 1
        else:
            self.commit_interval = max(0, interval)

        self._run_id: str | None = None
        self._event_seq: int = 0
        self._trace_rows: int = 0
        self._conn: Any | None = None
        self._schema_ready: bool = False
        self._pending_writes: int = 0

    @staticmethod
    def _normalize_io_mode(value: Any) -> str:
        mode = str(value or "batched").strip().lower()
        if mode not in {"legacy", "batched", "safe"}:
            raise ValueError("experiment_tracker io_mode must be one of: legacy, batched, safe")
        return mode

    def _connect(self) -> Any:
        return open_experiment_db(self.db_path)

    def _open_session(self) -> Any:
        if self._conn is not None:
            return self._conn
        self._conn = self._connect()
        self._schema_ready = False
        self._pending_writes = 0
        return self._conn

    def _close_session(self, *, commit: bool) -> None:
        conn = self._conn
        self._conn = None
        self._schema_ready = False
        self._pending_writes = 0
        if conn is None:
            return
        try:
            if commit:
                conn.commit()
        finally:
            conn.close()

    def close(self) -> None:
        self._close_session(commit=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_schema_once(self, conn: Any) -> None:
        if self._schema_ready:
            return
        self._ensure_schema(conn)
        self._schema_ready = True

    def _maybe_commit(self, *, force: bool = False) -> None:
        conn = self._conn
        if conn is None:
            return
        if force:
            conn.commit()
            self._pending_writes = 0
            return
        if self.commit_interval <= 0:
            return
        if self._pending_writes >= self.commit_interval:
            conn.commit()
            self._pending_writes = 0

    def _write(
        self,
        writer: Callable[[Any], None],
        *,
        ensure_schema: bool = False,
        force_commit: bool = False,
    ) -> None:
        if self.io_mode == "legacy":
            with self._connect() as conn:
                if ensure_schema:
                    self._ensure_schema(conn)
                writer(conn)
                conn.commit()
            return

        conn = self._open_session()
        if ensure_schema:
            self._ensure_schema_once(conn)
        writer(conn)
        self._pending_writes += 1
        self._maybe_commit(force=force_commit)

    def _ensure_schema(self, conn: Any) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                tag TEXT,
                run_name TEXT,
                started_at_utc TEXT NOT NULL,
                finished_at_utc TEXT,
                status TEXT NOT NULL,
                trainer_name TEXT,
                output_dir TEXT,
                metrics_json TEXT,
                report_json TEXT,
                model_spec_json TEXT,
                error_text TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts_utc TEXT NOT NULL,
                event TEXT NOT NULL,
                stage TEXT,
                payload_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts_utc TEXT NOT NULL,
                split TEXT NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_training_trace (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                ts_utc TEXT NOT NULL,
                operation TEXT,
                selected_name TEXT,
                selected_family TEXT,
                selected_expr TEXT,
                n_terms_before INTEGER,
                n_terms_after INTEGER,
                rmse_before REAL,
                rmse_after REAL,
                val_rmse_before REAL,
                val_rmse_after REAL,
                grad_overall_mismatch REAL,
                weight_l2_before REAL,
                weight_l2_after REAL,
                payload_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_run_catalog (
                run_id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                tag TEXT,
                run_name TEXT,
                status TEXT NOT NULL,
                trainer_name TEXT,
                output_dir TEXT,
                artifact_id TEXT,
                training_mode TEXT,
                surface_key TEXT,
                surface_kind TEXT,
                surface_signature TEXT,
                assembly_signature TEXT,
                family_ref TEXT,
                driver_ref TEXT,
                symbolic_family_signature TEXT,
                task_signature_json TEXT,
                symbolic_family_json TEXT,
                search_mechanism_contracts_json TEXT,
                search_family_signature_contracts_json TEXT,
                search_mechanism_keys_json TEXT,
                search_family_signature_keys_json TEXT,
                compatibility_json TEXT,
                compatibility_drift_json TEXT,
                regime_mode TEXT,
                basis_scope TEXT,
                assembler_mode TEXT,
                piecewise_gate_status TEXT,
                orthogonality_status TEXT,
                orthogonality_score REAL,
                pair_abs_corr_mean REAL,
                residual_complementarity_status TEXT,
                residual_gain_mean REAL,
                semantic_dedup_status TEXT,
                semantic_unique_ratio REAL,
                gate_basis_count INTEGER,
                selected_regime_count INTEGER,
                basis_count INTEGER,
                output_expression_count INTEGER,
                regime_structure_json TEXT,
                basis_structure_json TEXT,
                assembler_structure_json TEXT,
                piecewise_gate_basis_json TEXT,
                fold_count INTEGER,
                fold_summary_json TEXT,
                rmse_mean REAL,
                rmse_std REAL,
                rmse_drift REAL,
                coverage_error_mean REAL,
                pinaw_mean REAL,
                interval_score_mean REAL,
                picp_mean REAL,
                mean_width_mean REAL,
                family_concentration REAL,
                feature_concentration REAL,
                exact_basis_hit_score REAL,
                exact_term_recovery_score REAL,
                outer_objective_score REAL,
                inner_fit_score REAL,
                truth_contract_recovery_json TEXT,
                orthogonal_search_objective_json TEXT,
                artifact_catalog_json TEXT,
                report_json TEXT,
                surface_record_json TEXT,
                assembly_record_json TEXT,
                run_record_json TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_artifact_catalog (
                run_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                trainer_name TEXT,
                artifact_kind TEXT,
                head_task TEXT,
                head_semantics_json TEXT,
                complexity_metrics_json TEXT,
                stability_metrics_json TEXT,
                regime_mode TEXT,
                basis_scope TEXT,
                assembler_mode TEXT,
                piecewise_gate_status TEXT,
                orthogonality_status TEXT,
                orthogonality_score REAL,
                pair_abs_corr_mean REAL,
                residual_complementarity_status TEXT,
                residual_gain_mean REAL,
                semantic_dedup_status TEXT,
                semantic_unique_ratio REAL,
                gate_basis_count INTEGER,
                selected_regime_count INTEGER,
                basis_count INTEGER,
                output_expression_count INTEGER,
                regime_structure_json TEXT,
                basis_structure_json TEXT,
                assembler_structure_json TEXT,
                piecewise_gate_basis_json TEXT,
                fold_summary_json TEXT,
                symbolic_family_signature TEXT,
                search_family_signature_contracts_json TEXT,
                fold_count INTEGER,
                rmse_mean REAL,
                rmse_std REAL,
                rmse_drift REAL,
                coverage_error_mean REAL,
                pinaw_mean REAL,
                interval_score_mean REAL,
                picp_mean REAL,
                mean_width_mean REAL,
                family_concentration REAL,
                feature_concentration REAL,
                exact_basis_hit_score REAL,
                exact_term_recovery_score REAL,
                outer_objective_score REAL,
                inner_fit_score REAL,
                truth_contract_recovery_json TEXT,
                orthogonal_search_objective_json TEXT,
                source_report_json TEXT,
                artifact_record_json TEXT,
                PRIMARY KEY (run_id, artifact_id)
            )
            """
        )
        ensure_table_columns(
            conn,
            "experiment_run_catalog",
            {
                "surface_key": "TEXT",
                "surface_kind": "TEXT",
                "surface_signature": "TEXT",
                "assembly_signature": "TEXT",
                "family_ref": "TEXT",
                "driver_ref": "TEXT",
                "regime_mode": "TEXT",
                "basis_scope": "TEXT",
                "assembler_mode": "TEXT",
                "piecewise_gate_status": "TEXT",
                "orthogonality_status": "TEXT",
                "orthogonality_score": "REAL",
                "pair_abs_corr_mean": "REAL",
                "residual_complementarity_status": "TEXT",
                "residual_gain_mean": "REAL",
                "semantic_dedup_status": "TEXT",
                "semantic_unique_ratio": "REAL",
                "gate_basis_count": "INTEGER",
                "selected_regime_count": "INTEGER",
                "basis_count": "INTEGER",
                "output_expression_count": "INTEGER",
                "regime_structure_json": "TEXT",
                "basis_structure_json": "TEXT",
                "assembler_structure_json": "TEXT",
                "piecewise_gate_basis_json": "TEXT",
                "exact_basis_hit_score": "REAL",
                "exact_term_recovery_score": "REAL",
                "outer_objective_score": "REAL",
                "inner_fit_score": "REAL",
                "truth_contract_recovery_json": "TEXT",
                "orthogonal_search_objective_json": "TEXT",
                "surface_record_json": "TEXT",
                "assembly_record_json": "TEXT",
                "run_record_json": "TEXT",
            },
        )
        ensure_table_columns(
            conn,
            "experiment_artifact_catalog",
            {
                "artifact_record_json": "TEXT",
                "regime_mode": "TEXT",
                "basis_scope": "TEXT",
                "assembler_mode": "TEXT",
                "piecewise_gate_status": "TEXT",
                "orthogonality_status": "TEXT",
                "orthogonality_score": "REAL",
                "pair_abs_corr_mean": "REAL",
                "residual_complementarity_status": "TEXT",
                "residual_gain_mean": "REAL",
                "semantic_dedup_status": "TEXT",
                "semantic_unique_ratio": "REAL",
                "gate_basis_count": "INTEGER",
                "selected_regime_count": "INTEGER",
                "basis_count": "INTEGER",
                "output_expression_count": "INTEGER",
                "regime_structure_json": "TEXT",
                "basis_structure_json": "TEXT",
                "assembler_structure_json": "TEXT",
                "piecewise_gate_basis_json": "TEXT",
                "exact_basis_hit_score": "REAL",
                "exact_term_recovery_score": "REAL",
                "outer_objective_score": "REAL",
                "inner_fit_score": "REAL",
                "truth_contract_recovery_json": "TEXT",
                "orthogonal_search_objective_json": "TEXT",
            },
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_events_run_id ON experiment_events(run_id, seq)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_metrics_run_id ON experiment_metrics(run_id, split, metric)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_trace_run_id ON experiment_training_trace(run_id, iteration)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_status ON experiment_run_catalog(status, trainer_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_surface_key ON experiment_run_catalog(surface_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_assembly_signature ON experiment_run_catalog(assembly_signature)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_family_ref ON experiment_run_catalog(family_ref)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_regime_mode ON experiment_run_catalog(regime_mode)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_basis_scope ON experiment_run_catalog(basis_scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_assembler_mode ON experiment_run_catalog(assembler_mode)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_piecewise_gate_status ON experiment_run_catalog(piecewise_gate_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_orthogonality_status ON experiment_run_catalog(orthogonality_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_semantic_dedup_status ON experiment_run_catalog(semantic_dedup_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_rmse_std ON experiment_run_catalog(rmse_std)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_exact_basis_hit_score ON experiment_run_catalog(exact_basis_hit_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_exact_term_recovery_score ON experiment_run_catalog(exact_term_recovery_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_outer_objective_score ON experiment_run_catalog(outer_objective_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_run_catalog_coverage_error ON experiment_run_catalog(coverage_error_mean)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_rmse_std ON experiment_artifact_catalog(rmse_std)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_regime_mode ON experiment_artifact_catalog(regime_mode)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_basis_scope ON experiment_artifact_catalog(basis_scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_assembler_mode ON experiment_artifact_catalog(assembler_mode)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_piecewise_gate_status ON experiment_artifact_catalog(piecewise_gate_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_orthogonality_status ON experiment_artifact_catalog(orthogonality_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_semantic_dedup_status ON experiment_artifact_catalog(semantic_dedup_status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_exact_basis_hit_score ON experiment_artifact_catalog(exact_basis_hit_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_exact_term_recovery_score ON experiment_artifact_catalog(exact_term_recovery_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_outer_objective_score ON experiment_artifact_catalog(outer_objective_score)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiment_artifact_catalog_coverage_error ON experiment_artifact_catalog(coverage_error_mean)"
        )

    def _new_run_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:10]
        return f"{self.namespace}_{ts}_{suffix}"

    def _payload_to_json(self, payload: Mapping[str, Any] | None) -> str:
        raw = _to_jsonable(dict(payload or {}))
        text = json.dumps(raw, ensure_ascii=False)
        if len(text) <= self.max_payload_chars:
            return text
        clipped = {
            "_truncated": True,
            "_max_payload_chars": int(self.max_payload_chars),
            "preview": text[: self.max_payload_chars],
        }
        return json.dumps(clipped, ensure_ascii=False)

    def _safe_context_summary(self, context: MutableMapping[str, Any]) -> Dict[str, Any]:
        refs_raw = context.get("context_refs", {})
        refs = refs_raw if isinstance(refs_raw, Mapping) else {}
        return {
            "run_name": context.get("run_name"),
            "stage": refs.get(RUN_STAGE),
            "snapshot_count": context.get("snapshot_count"),
        }

    def _append_event(
        self,
        conn: Any,
        *,
        event: str,
        context: MutableMapping[str, Any],
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self._run_id is None:
            return
        self._event_seq += 1
        base = self._safe_context_summary(context)
        merged = dict(base)
        if payload:
            merged.update(dict(payload))
        conn.execute(
            """
            INSERT INTO experiment_events (run_id, seq, ts_utc, event, stage, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(self._run_id),
                int(self._event_seq),
                _utc_now_iso(),
                str(event),
                None if merged.get("stage") is None else str(merged.get("stage")),
                self._payload_to_json(merged),
            ),
        )

    def _set_report_info(self, context: MutableMapping[str, Any], *, status: str = "running") -> None:
        info = {
            "run_id": self._run_id,
            "db_path": self.db_path,
            "namespace": self.namespace,
            "tag": self.tag,
            "status": str(status),
            "trace_rows": int(self._trace_rows),
        }
        context[self.report_key] = info
        report_raw = context.get("report")
        if isinstance(report_raw, dict):
            report_raw[self.report_key] = dict(info)

    @staticmethod
    def _as_float_or_none(value: Any) -> float | None:
        try:
            f = float(value)
        except Exception:
            return None
        if not math.isfinite(f):
            return None
        return float(f)

    @staticmethod
    def _as_int_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except Exception:
            return None

    @staticmethod
    def _mapping(value: Any) -> dict[str, Any]:
        return {str(k): v for k, v in dict(value).items()} if isinstance(value, Mapping) else {}

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _catalog_ref(prefix: str, value: Any) -> str | None:
        text = ExperimentTrackerCapability._text(value)
        if text is None:
            return None
        return text if ":" in text else f"{prefix}:{text}"

    @staticmethod
    def _family_ref_from_driver(trainer_name: str | None, task_signature: Mapping[str, Any], symbolic_family: Mapping[str, Any]) -> str | None:
        explicit = ExperimentTrackerCapability._text(task_signature.get("family"))
        if explicit is None and isinstance(task_signature.get("family_ref"), str):
            explicit = ExperimentTrackerCapability._text(task_signature.get("family_ref"))
        if explicit:
            return explicit if explicit.startswith("family:") else f"family:{explicit}"
        if symbolic_family or task_signature.get("symbolic_family_signature") is not None:
            return "family:symbolic"
        name = str(trainer_name or "").strip().lower()
        if not name:
            return None
        if "symbolic" in name:
            return "family:symbolic"
        if "ridge" in name or name == "linear":
            return "family:linear"
        if "xgboost" in name or "boost" in name:
            return "family:tree_boosting"
        if "forest" in name or "tree" in name or "adaboost" in name or "bagging" in name:
            return "family:tree_ensemble"
        if "mlp" in name or "torch" in name or "neural" in name:
            return "family:neural"
        return None

    @staticmethod
    def _coerce_ref_tuple(values: Sequence[Any] | None, prefix: str = "") -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for item in values or ():
            text = ExperimentTrackerCapability._text(item)
            if text is None:
                continue
            ref = text if (not prefix or ":" in text) else f"{prefix}:{text}"
            if ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
        return tuple(out)

    @staticmethod
    def _metric_first(metrics: Mapping[str, Any]) -> tuple[str | None, float | None]:
        preferred = (
            "rmse_mean",
            "rmse",
            "mae",
            "coverage_error_mean",
            "r2",
        )
        for key in preferred:
            if key not in metrics:
                continue
            try:
                return str(key), float(metrics[key])
            except Exception:
                continue
        for key, value in metrics.items():
            try:
                return str(key), float(value)
            except Exception:
                continue
        return None, None

    def _build_contract_records(
        self,
        context: MutableMapping[str, Any],
        *,
        report: Mapping[str, Any],
        training: Mapping[str, Any],
        artifact_block: Mapping[str, Any],
        task_signature: Mapping[str, Any],
        symbolic_family: Mapping[str, Any],
        head_semantics: Mapping[str, Any],
        stability: Mapping[str, Any],
        fold_summary: Mapping[str, Any],
        trainer_name: str | None,
        artifact_id: str | None,
        started_at_utc: str | None,
        finished_at_utc: str | None,
        status: str,
    ) -> tuple[SurfaceRecord, AssemblyRecord, ArtifactRecord | None, RunRecord]:
        framework = "mlblack"
        run_name = str(context.get("run_name", report.get("run_name", "train_flow")))
        surface_kind = str(context.get("surface_kind", "flow"))
        surface_key = (
            self._text(context.get("surface_key"))
            or self._text(context.get("flow_key"))
            or self._text(report.get("flow_key"))
            or f"{surface_kind}:{run_name}"
        )
        driver_ref = self._catalog_ref("trainer", trainer_name)
        family_ref = self._family_ref_from_driver(trainer_name, task_signature, symbolic_family)
        project_root = self._text(context.get("project_root"))
        scaffold_root = self._text(context.get("scaffold_root")) or project_root
        entry_path = self._text(context.get("entry_path"))
        entry_module = self._text(context.get("entry_module"))
        entry_symbol = self._text(context.get("entry_symbol"))

        surface_record = make_surface_record(
            framework=framework,
            project_root=project_root,
            scaffold_root=scaffold_root,
            surface_kind=surface_kind,
            surface_key=str(surface_key),
            surface_label=run_name,
            entry_path=entry_path,
            entry_module=entry_module,
            entry_symbol=entry_symbol,
            driver_ref=driver_ref,
            family_ref=family_ref,
            tags=("experiment_surface", surface_kind),
            metadata_json={
                "run_name": run_name,
                "namespace": self.namespace,
            },
        )

        head_ref = self._catalog_ref("head", head_semantics.get("task"))
        preset_ref = self._catalog_ref("preset", context.get("preset_key")) or self._catalog_ref("preset", trainer_name)
        component_refs = self._coerce_ref_tuple(training.get("component_refs"), prefix="component")
        provider_refs = self._coerce_ref_tuple(training.get("provider_refs"), prefix="provider")
        plugin_refs = self._coerce_ref_tuple(training.get("plugin_refs"), prefix="plugin")
        if "plugin:experiment_tracker" not in plugin_refs:
            plugin_refs = tuple(list(plugin_refs) + ["plugin:experiment_tracker"])
        mount_order = tuple(
            ref
            for ref in (
                family_ref,
                preset_ref,
                head_ref,
                *component_refs,
                *provider_refs,
                *plugin_refs,
            )
            if ref
        )
        assembly_record = make_assembly_record(
            framework=framework,
            surface_key=str(surface_key),
            assembly_key=self._text(context.get("assembly_key")) or self._text(report.get("assembly_key")) or str(surface_key),
            driver_ref=driver_ref,
            family_ref=family_ref,
            preset_ref=preset_ref,
            head_ref=head_ref,
            trainer_ref=driver_ref,
            component_refs=component_refs,
            provider_refs=provider_refs,
            plugin_refs=plugin_refs,
            mount_order=mount_order,
            component_slots_json={
                "family": family_ref,
                "preset": preset_ref,
                "head": head_ref,
                "components": list(component_refs),
                "providers": list(provider_refs),
                "plugins": list(plugin_refs),
            },
            metadata_json={
                "training_mode": self._mapping(training.get("requested_init")).get("mode"),
                "search_mechanism_keys": list(symbolic_family.get("search_mechanism_keys", ()) or ()),
            },
        )

        subject_json = _to_jsonable(context.get("model_spec"))
        if not subject_json:
            subject_json = _to_jsonable(task_signature)
        metric_summary = dict(_to_jsonable(fold_summary or stability or context.get("metrics") or {}))
        primary_metric_name, primary_metric_value = self._metric_first(metric_summary)
        params_json = {
            "requested_init": _to_jsonable(training.get("requested_init")),
            "task_signature": _to_jsonable(task_signature),
        }
        duration_s = None
        if started_at_utc and finished_at_utc:
            try:
                duration_s = max(
                    0.0,
                    float(
                        (
                            datetime.fromisoformat(str(finished_at_utc))
                            - datetime.fromisoformat(str(started_at_utc))
                        ).total_seconds()
                    ),
                )
            except Exception:
                duration_s = None
        run_record = make_run_record(
            framework=framework,
            run_id=str(self._run_id or ""),
            namespace=self.namespace,
            tag=self.tag,
            status=status,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            duration_s=duration_s,
            surface_key=str(surface_key),
            surface_kind=surface_kind,
            surface_signature=surface_record.surface_signature,
            assembly_signature=assembly_record.assembly_signature,
            subject_kind="model_spec" if context.get("model_spec") is not None else "task_signature",
            subject_key=self._text(getattr(context.get("model_spec"), "model_id", None)) or run_name,
            subject_json=subject_json if isinstance(subject_json, Mapping) else {"value": subject_json},
            params_json=params_json,
            driver_ref=driver_ref,
            family_ref=family_ref,
            output_dir=self._text(context.get("output_dir")),
            primary_metric_name=primary_metric_name,
            primary_metric_value=primary_metric_value,
            metric_summary_json=metric_summary,
            result_json=_to_jsonable(report) if isinstance(report, Mapping) else {},
            component_refs=tuple(
                ref
                for ref in (
                    family_ref,
                    preset_ref,
                    head_ref,
                    *component_refs,
                    *provider_refs,
                    *plugin_refs,
                )
                if ref
            ),
            artifact_ids=(artifact_id,) if artifact_id else (),
            metadata_json={
                "trainer_name": trainer_name,
                "surface_record": surface_record.to_dict(),
                "assembly_record": assembly_record.to_dict(),
            },
        )

        artifact_record: ArtifactRecord | None = None
        if artifact_id:
            artifact_record = make_artifact_record(
                framework=framework,
                run_id=str(self._run_id or ""),
                artifact_id=str(artifact_id),
                artifact_kind=str(artifact_block.get("artifact_kind", "surrogate_artifact")),
                artifact_role="primary_model_artifact",
                producer_ref=preset_ref or driver_ref,
                surface_key=str(surface_key),
                assembly_signature=assembly_record.assembly_signature,
                path=self._text(artifact_block.get("artifact_path")),
                uri=self._text(artifact_block.get("artifact_uri")),
                format=self._text(artifact_block.get("artifact_format")),
                created_at_utc=finished_at_utc,
                metrics_json=metric_summary,
                metadata_json={
                    "head_semantics": _to_jsonable(head_semantics),
                    "complexity_metrics": _to_jsonable(self._mapping(self._mapping(artifact_block.get("symbolic_artifact_schema")).get("complexity_metrics"))),
                    "stability_metrics": _to_jsonable(stability),
                },
                tags=("artifact", str(head_semantics.get("task") or "unknown")),
            )
        return surface_record, assembly_record, artifact_record, run_record

    def _extract_report_catalog_payloads(
        self,
        context: MutableMapping[str, Any],
        *,
        status: str,
        started_at_utc: str | None,
        finished_at_utc: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        report = self._mapping(context.get("report"))
        training = self._mapping(report.get("training"))
        artifact_block = self._mapping(report.get("artifact"))
        task_signature = self._mapping(training.get("task_signature"))
        if not task_signature:
            fit_report = self._mapping(training.get("fit_report"))
            task_signature = self._mapping(fit_report.get("task_signature"))
        signature_meta = self._mapping(task_signature.get("metadata"))
        symbolic_family = self._mapping(signature_meta.get("symbolic_family"))
        compatibility = self._mapping(training.get("compatibility"))
        if not compatibility:
            fit_report = self._mapping(training.get("fit_report"))
            compatibility = self._mapping(fit_report.get("compatibility"))
        compatibility_drift = self._mapping(training.get("symbolic_family_signature_drift"))
        if not compatibility_drift:
            compatibility_drift = self._mapping(compatibility.get("symbolic_family_signature_drift"))

        symbolic_schema = self._mapping(artifact_block.get("symbolic_artifact_schema"))
        complexity = self._mapping(symbolic_schema.get("complexity_metrics"))
        stability = self._mapping(symbolic_schema.get("stability_metrics"))
        head_semantics = self._mapping(symbolic_schema.get("head_semantics"))
        regime_structure = self._mapping(symbolic_schema.get("regime_structure"))
        basis_structure = self._mapping(symbolic_schema.get("basis_structure"))
        assembler_structure = self._mapping(symbolic_schema.get("assembler_structure"))
        piecewise_gate_basis = self._mapping(symbolic_schema.get("piecewise_gate_basis"))
        truth_contract_recovery = self._mapping(symbolic_schema.get("truth_contract_recovery"))
        orthogonal_search_objective = self._mapping(symbolic_schema.get("orthogonal_search_objective"))
        orthogonality = self._mapping(basis_structure.get("orthogonality_status"))
        residual_complementarity = self._mapping(basis_structure.get("residual_complementarity"))
        semantic_deduplication = self._mapping(basis_structure.get("semantic_deduplication"))
        fold_summary = self._mapping(stability.get("fold_summary"))

        mechanism_contracts = tuple(symbolic_family.get("search_mechanism_contracts", ()))
        signature_contracts = tuple(symbolic_family.get("search_family_signature_contracts", ()))
        search_mechanism_keys = tuple(
            sorted(
                str(self._mapping(row).get("mechanism_key", "")).strip()
                for row in mechanism_contracts
                if str(self._mapping(row).get("mechanism_key", "")).strip()
            )
        )
        search_family_signature_keys = tuple(
            sorted(
                str(self._mapping(row).get("mechanism_key", "")).strip()
                for row in signature_contracts
                if str(self._mapping(row).get("mechanism_key", "")).strip()
            )
        )

        artifact_id = str(artifact_block.get("artifact_id", getattr(context.get("artifact"), "artifact_id", "")) or "")
        trainer = context.get("trainer")
        trainer_name = None if trainer is None else str(getattr(trainer, "name", type(trainer).__name__))
        if trainer_name is None:
            trainer_name = None if report.get("trainer_name") is None else str(report.get("trainer_name"))

        surface_record, assembly_record, artifact_record, run_record = self._build_contract_records(
            context,
            report=report,
            training=training,
            artifact_block=artifact_block,
            task_signature=task_signature,
            symbolic_family=symbolic_family,
            head_semantics=head_semantics,
            stability=stability,
            fold_summary=fold_summary,
            trainer_name=trainer_name,
            artifact_id=artifact_id or None,
            started_at_utc=started_at_utc,
            finished_at_utc=finished_at_utc,
            status=str(status),
        )

        run_catalog = {
            "run_id": None if self._run_id is None else str(self._run_id),
            "namespace": str(self.namespace),
            "tag": self.tag,
            "run_name": str(context.get("run_name", report.get("run_name", "train_flow"))),
            "status": str(status),
            "trainer_name": trainer_name,
            "output_dir": None if context.get("output_dir") is None else str(context.get("output_dir")),
            "artifact_id": artifact_id or None,
            "training_mode": (
                str(training.get("requested_init", {}).get("mode"))
                if isinstance(training.get("requested_init"), Mapping)
                else None
            )
            or None,
            "surface_key": surface_record.surface_key,
            "surface_kind": surface_record.surface_kind,
            "surface_signature": surface_record.surface_signature,
            "assembly_signature": assembly_record.assembly_signature,
            "family_ref": run_record.family_ref,
            "driver_ref": run_record.driver_ref,
            "symbolic_family_signature": None
            if task_signature.get("symbolic_family_signature") is None
            else str(task_signature.get("symbolic_family_signature")),
            "task_signature_json": dict(task_signature) or None,
            "symbolic_family_json": dict(symbolic_family) or None,
            "search_mechanism_contracts_json": list(mechanism_contracts),
            "search_family_signature_contracts_json": list(signature_contracts),
            "search_mechanism_keys_json": list(search_mechanism_keys),
            "search_family_signature_keys_json": list(search_family_signature_keys),
            "compatibility_json": dict(compatibility) or None,
            "compatibility_drift_json": dict(compatibility_drift) or None,
            "regime_mode": self._text(regime_structure.get("mode")),
            "basis_scope": self._text(basis_structure.get("basis_scope")),
            "assembler_mode": self._text(assembler_structure.get("assembler_mode")),
            "piecewise_gate_status": self._text(piecewise_gate_basis.get("status")),
            "orthogonality_status": self._text(orthogonality.get("status")),
            "orthogonality_score": self._as_float_or_none(orthogonality.get("orthogonality_score")),
            "pair_abs_corr_mean": self._as_float_or_none(orthogonality.get("pair_abs_corr_mean")),
            "residual_complementarity_status": self._text(residual_complementarity.get("status")),
            "residual_gain_mean": self._as_float_or_none(
                self._mapping(residual_complementarity.get("recorded")).get("mean_marginal_r2_gain")
            ),
            "semantic_dedup_status": self._text(semantic_deduplication.get("status")),
            "semantic_unique_ratio": self._as_float_or_none(
                self._mapping(semantic_deduplication.get("recorded")).get("semantic_unique_ratio")
            ),
            "gate_basis_count": self._as_int_or_none(piecewise_gate_basis.get("gate_basis_count")),
            "selected_regime_count": self._as_int_or_none(regime_structure.get("local_regime_count")),
            "basis_count": self._as_int_or_none(basis_structure.get("basis_count")),
            "output_expression_count": self._as_int_or_none(assembler_structure.get("output_expression_count")),
            "regime_structure_json": dict(regime_structure) or None,
            "basis_structure_json": dict(basis_structure) or None,
            "assembler_structure_json": dict(assembler_structure) or None,
            "piecewise_gate_basis_json": dict(piecewise_gate_basis) or None,
            "fold_count": self._as_int_or_none(stability.get("fold_count")),
            "fold_summary_json": dict(fold_summary) or None,
            "rmse_mean": self._as_float_or_none(stability.get("rmse_mean")),
            "rmse_std": self._as_float_or_none(stability.get("rmse_std")),
            "rmse_drift": self._as_float_or_none(stability.get("rmse_drift")),
            "coverage_error_mean": self._as_float_or_none(stability.get("coverage_error_mean")),
            "pinaw_mean": self._as_float_or_none(stability.get("pinaw_mean")),
            "interval_score_mean": self._as_float_or_none(stability.get("interval_score_mean")),
            "picp_mean": self._as_float_or_none(stability.get("picp_mean")),
            "mean_width_mean": self._as_float_or_none(stability.get("mean_width_mean")),
            "family_concentration": self._as_float_or_none(stability.get("family_concentration")),
            "feature_concentration": self._as_float_or_none(stability.get("feature_concentration")),
            "exact_basis_hit_score": self._as_float_or_none(truth_contract_recovery.get("exact_basis_hit_score")),
            "exact_term_recovery_score": self._as_float_or_none(truth_contract_recovery.get("exact_term_recovery_score")),
            "outer_objective_score": self._as_float_or_none(orthogonal_search_objective.get("outer_score")),
            "inner_fit_score": self._as_float_or_none(orthogonal_search_objective.get("inner_fit_score")),
            "truth_contract_recovery_json": dict(truth_contract_recovery) or None,
            "orthogonal_search_objective_json": dict(orthogonal_search_objective) or None,
            "artifact_catalog_json": {
                "artifact_id": artifact_id or None,
                "head_semantics": dict(head_semantics) or None,
                "complexity_metrics": dict(complexity) or None,
                "stability_metrics": dict(stability) or None,
                "fold_summary": dict(fold_summary) or None,
                "regime_structure": dict(regime_structure) or None,
                "basis_structure": dict(basis_structure) or None,
                "assembler_structure": dict(assembler_structure) or None,
                "piecewise_gate_basis": dict(piecewise_gate_basis) or None,
                "truth_contract_recovery": dict(truth_contract_recovery) or None,
                "orthogonal_search_objective": dict(orthogonal_search_objective) or None,
            },
            "report_json": dict(report) or None,
            "surface_record_json": surface_record.to_dict(),
            "assembly_record_json": assembly_record.to_dict(),
            "run_record_json": run_record.to_dict(),
        }
        artifact_catalog = {
            "run_id": None if self._run_id is None else str(self._run_id),
            "artifact_id": artifact_id or None,
            "trainer_name": trainer_name,
            "artifact_kind": str(artifact_block.get("artifact_kind", "surrogate_artifact")),
            "head_task": None if head_semantics.get("task") is None else str(head_semantics.get("task")),
            "head_semantics_json": dict(head_semantics) or None,
            "complexity_metrics_json": dict(complexity) or None,
            "stability_metrics_json": dict(stability) or None,
            "regime_mode": self._text(regime_structure.get("mode")),
            "basis_scope": self._text(basis_structure.get("basis_scope")),
            "assembler_mode": self._text(assembler_structure.get("assembler_mode")),
            "piecewise_gate_status": self._text(piecewise_gate_basis.get("status")),
            "orthogonality_status": self._text(orthogonality.get("status")),
            "orthogonality_score": self._as_float_or_none(orthogonality.get("orthogonality_score")),
            "pair_abs_corr_mean": self._as_float_or_none(orthogonality.get("pair_abs_corr_mean")),
            "residual_complementarity_status": self._text(residual_complementarity.get("status")),
            "residual_gain_mean": self._as_float_or_none(
                self._mapping(residual_complementarity.get("recorded")).get("mean_marginal_r2_gain")
            ),
            "semantic_dedup_status": self._text(semantic_deduplication.get("status")),
            "semantic_unique_ratio": self._as_float_or_none(
                self._mapping(semantic_deduplication.get("recorded")).get("semantic_unique_ratio")
            ),
            "gate_basis_count": self._as_int_or_none(piecewise_gate_basis.get("gate_basis_count")),
            "selected_regime_count": self._as_int_or_none(regime_structure.get("local_regime_count")),
            "basis_count": self._as_int_or_none(basis_structure.get("basis_count")),
            "output_expression_count": self._as_int_or_none(assembler_structure.get("output_expression_count")),
            "regime_structure_json": dict(regime_structure) or None,
            "basis_structure_json": dict(basis_structure) or None,
            "assembler_structure_json": dict(assembler_structure) or None,
            "piecewise_gate_basis_json": dict(piecewise_gate_basis) or None,
            "fold_summary_json": dict(fold_summary) or None,
            "symbolic_family_signature": None
            if task_signature.get("symbolic_family_signature") is None
            else str(task_signature.get("symbolic_family_signature")),
            "search_family_signature_contracts_json": list(signature_contracts),
            "fold_count": self._as_int_or_none(stability.get("fold_count")),
            "rmse_mean": self._as_float_or_none(stability.get("rmse_mean")),
            "rmse_std": self._as_float_or_none(stability.get("rmse_std")),
            "rmse_drift": self._as_float_or_none(stability.get("rmse_drift")),
            "coverage_error_mean": self._as_float_or_none(stability.get("coverage_error_mean")),
            "pinaw_mean": self._as_float_or_none(stability.get("pinaw_mean")),
            "interval_score_mean": self._as_float_or_none(stability.get("interval_score_mean")),
            "picp_mean": self._as_float_or_none(stability.get("picp_mean")),
            "mean_width_mean": self._as_float_or_none(stability.get("mean_width_mean")),
            "family_concentration": self._as_float_or_none(stability.get("family_concentration")),
            "feature_concentration": self._as_float_or_none(stability.get("feature_concentration")),
            "exact_basis_hit_score": self._as_float_or_none(truth_contract_recovery.get("exact_basis_hit_score")),
            "exact_term_recovery_score": self._as_float_or_none(truth_contract_recovery.get("exact_term_recovery_score")),
            "outer_objective_score": self._as_float_or_none(orthogonal_search_objective.get("outer_score")),
            "inner_fit_score": self._as_float_or_none(orthogonal_search_objective.get("inner_fit_score")),
            "truth_contract_recovery_json": dict(truth_contract_recovery) or None,
            "orthogonal_search_objective_json": dict(orthogonal_search_objective) or None,
            "source_report_json": dict(report) or None,
            "artifact_record_json": None if artifact_record is None else artifact_record.to_dict(),
        }
        return run_catalog, artifact_catalog

    def _upsert_catalog_rows(
        self,
        conn: Any,
        context: MutableMapping[str, Any],
        *,
        status: str,
        finished_at_utc: str | None = None,
    ) -> None:
        if self._run_id is None:
            return
        started_row = conn.execute(
            "SELECT started_at_utc FROM experiment_runs WHERE run_id = ?",
            (str(self._run_id),),
        ).fetchone()
        started_at_utc = None if started_row is None else str(started_row[0] or "")
        run_catalog, artifact_catalog = self._extract_report_catalog_payloads(
            context,
            status=str(status),
            started_at_utc=started_at_utc or None,
            finished_at_utc=finished_at_utc,
        )
        run_catalog["status"] = str(status)
        if run_catalog.get("run_id") is None:
            run_catalog["run_id"] = str(self._run_id)
        if artifact_catalog.get("run_id") is None:
            artifact_catalog["run_id"] = str(self._run_id)

        run_insert_columns = (
            "run_id",
            "namespace",
            "tag",
            "run_name",
            "status",
            "trainer_name",
            "output_dir",
            "artifact_id",
            "training_mode",
            "surface_key",
            "surface_kind",
            "surface_signature",
            "assembly_signature",
            "family_ref",
            "driver_ref",
            "symbolic_family_signature",
            "task_signature_json",
            "symbolic_family_json",
            "search_mechanism_contracts_json",
            "search_family_signature_contracts_json",
            "search_mechanism_keys_json",
            "search_family_signature_keys_json",
            "compatibility_json",
            "compatibility_drift_json",
            "regime_mode",
            "basis_scope",
            "assembler_mode",
            "piecewise_gate_status",
            "orthogonality_status",
            "orthogonality_score",
            "pair_abs_corr_mean",
            "residual_complementarity_status",
            "residual_gain_mean",
            "semantic_dedup_status",
            "semantic_unique_ratio",
            "gate_basis_count",
            "selected_regime_count",
            "basis_count",
            "output_expression_count",
            "regime_structure_json",
            "basis_structure_json",
            "assembler_structure_json",
            "piecewise_gate_basis_json",
            "fold_count",
            "fold_summary_json",
            "rmse_mean",
            "rmse_std",
            "rmse_drift",
            "coverage_error_mean",
            "pinaw_mean",
            "interval_score_mean",
            "picp_mean",
            "mean_width_mean",
            "family_concentration",
            "feature_concentration",
            "exact_basis_hit_score",
            "exact_term_recovery_score",
            "outer_objective_score",
            "inner_fit_score",
            "truth_contract_recovery_json",
            "orthogonal_search_objective_json",
            "artifact_catalog_json",
            "report_json",
            "surface_record_json",
            "assembly_record_json",
            "run_record_json",
        )
        run_update_columns = tuple(name for name in run_insert_columns if name != "run_id")
        run_row = {
            "run_id": str(run_catalog.get("run_id")),
            "namespace": str(run_catalog.get("namespace", self.namespace)),
            "tag": run_catalog.get("tag"),
            "run_name": run_catalog.get("run_name"),
            "status": run_catalog.get("status"),
            "trainer_name": run_catalog.get("trainer_name"),
            "output_dir": run_catalog.get("output_dir"),
            "artifact_id": run_catalog.get("artifact_id"),
            "training_mode": run_catalog.get("training_mode"),
            "surface_key": run_catalog.get("surface_key"),
            "surface_kind": run_catalog.get("surface_kind"),
            "surface_signature": run_catalog.get("surface_signature"),
            "assembly_signature": run_catalog.get("assembly_signature"),
            "family_ref": run_catalog.get("family_ref"),
            "driver_ref": run_catalog.get("driver_ref"),
            "symbolic_family_signature": run_catalog.get("symbolic_family_signature"),
            "task_signature_json": None if run_catalog.get("task_signature_json") is None else _json_dumps(run_catalog.get("task_signature_json")),
            "symbolic_family_json": None if run_catalog.get("symbolic_family_json") is None else _json_dumps(run_catalog.get("symbolic_family_json")),
            "search_mechanism_contracts_json": (
                None
                if run_catalog.get("search_mechanism_contracts_json") is None
                else _json_dumps(run_catalog.get("search_mechanism_contracts_json"))
            ),
            "search_family_signature_contracts_json": (
                None
                if run_catalog.get("search_family_signature_contracts_json") is None
                else _json_dumps(run_catalog.get("search_family_signature_contracts_json"))
            ),
            "search_mechanism_keys_json": (
                None if run_catalog.get("search_mechanism_keys_json") is None else _json_dumps(run_catalog.get("search_mechanism_keys_json"))
            ),
            "search_family_signature_keys_json": (
                None
                if run_catalog.get("search_family_signature_keys_json") is None
                else _json_dumps(run_catalog.get("search_family_signature_keys_json"))
            ),
            "compatibility_json": None if run_catalog.get("compatibility_json") is None else _json_dumps(run_catalog.get("compatibility_json")),
            "compatibility_drift_json": (
                None
                if run_catalog.get("compatibility_drift_json") is None
                else _json_dumps(run_catalog.get("compatibility_drift_json"))
            ),
            "regime_mode": run_catalog.get("regime_mode"),
            "basis_scope": run_catalog.get("basis_scope"),
            "assembler_mode": run_catalog.get("assembler_mode"),
            "piecewise_gate_status": run_catalog.get("piecewise_gate_status"),
            "orthogonality_status": run_catalog.get("orthogonality_status"),
            "orthogonality_score": run_catalog.get("orthogonality_score"),
            "pair_abs_corr_mean": run_catalog.get("pair_abs_corr_mean"),
            "residual_complementarity_status": run_catalog.get("residual_complementarity_status"),
            "residual_gain_mean": run_catalog.get("residual_gain_mean"),
            "semantic_dedup_status": run_catalog.get("semantic_dedup_status"),
            "semantic_unique_ratio": run_catalog.get("semantic_unique_ratio"),
            "gate_basis_count": run_catalog.get("gate_basis_count"),
            "selected_regime_count": run_catalog.get("selected_regime_count"),
            "basis_count": run_catalog.get("basis_count"),
            "output_expression_count": run_catalog.get("output_expression_count"),
            "regime_structure_json": None if run_catalog.get("regime_structure_json") is None else _json_dumps(run_catalog.get("regime_structure_json")),
            "basis_structure_json": None if run_catalog.get("basis_structure_json") is None else _json_dumps(run_catalog.get("basis_structure_json")),
            "assembler_structure_json": (
                None if run_catalog.get("assembler_structure_json") is None else _json_dumps(run_catalog.get("assembler_structure_json"))
            ),
            "piecewise_gate_basis_json": (
                None
                if run_catalog.get("piecewise_gate_basis_json") is None
                else _json_dumps(run_catalog.get("piecewise_gate_basis_json"))
            ),
            "fold_count": run_catalog.get("fold_count"),
            "fold_summary_json": None if run_catalog.get("fold_summary_json") is None else _json_dumps(run_catalog.get("fold_summary_json")),
            "rmse_mean": run_catalog.get("rmse_mean"),
            "rmse_std": run_catalog.get("rmse_std"),
            "rmse_drift": run_catalog.get("rmse_drift"),
            "coverage_error_mean": run_catalog.get("coverage_error_mean"),
            "pinaw_mean": run_catalog.get("pinaw_mean"),
            "interval_score_mean": run_catalog.get("interval_score_mean"),
            "picp_mean": run_catalog.get("picp_mean"),
            "mean_width_mean": run_catalog.get("mean_width_mean"),
            "family_concentration": run_catalog.get("family_concentration"),
            "feature_concentration": run_catalog.get("feature_concentration"),
            "exact_basis_hit_score": run_catalog.get("exact_basis_hit_score"),
            "exact_term_recovery_score": run_catalog.get("exact_term_recovery_score"),
            "outer_objective_score": run_catalog.get("outer_objective_score"),
            "inner_fit_score": run_catalog.get("inner_fit_score"),
            "truth_contract_recovery_json": (
                None
                if run_catalog.get("truth_contract_recovery_json") is None
                else _json_dumps(run_catalog.get("truth_contract_recovery_json"))
            ),
            "orthogonal_search_objective_json": (
                None
                if run_catalog.get("orthogonal_search_objective_json") is None
                else _json_dumps(run_catalog.get("orthogonal_search_objective_json"))
            ),
            "artifact_catalog_json": None if run_catalog.get("artifact_catalog_json") is None else _json_dumps(run_catalog.get("artifact_catalog_json")),
            "report_json": None if run_catalog.get("report_json") is None else _json_dumps(run_catalog.get("report_json")),
            "surface_record_json": None if run_catalog.get("surface_record_json") is None else _json_dumps(run_catalog.get("surface_record_json")),
            "assembly_record_json": None if run_catalog.get("assembly_record_json") is None else _json_dumps(run_catalog.get("assembly_record_json")),
            "run_record_json": None if run_catalog.get("run_record_json") is None else _json_dumps(run_catalog.get("run_record_json")),
        }
        conn.execute(
            f"""
            INSERT INTO experiment_run_catalog ({", ".join(run_insert_columns)})
            VALUES ({", ".join(["?"] * len(run_insert_columns))})
            ON CONFLICT(run_id) DO UPDATE SET
                {", ".join(f"{name}=excluded.{name}" for name in run_update_columns)}
            """,
            tuple(run_row[name] for name in run_insert_columns),
        )

        artifact_id = artifact_catalog.get("artifact_id")
        if artifact_id:
            artifact_insert_columns = (
                "run_id",
                "artifact_id",
                "trainer_name",
                "artifact_kind",
                "head_task",
                "head_semantics_json",
                "complexity_metrics_json",
                "stability_metrics_json",
                "regime_mode",
                "basis_scope",
                "assembler_mode",
                "piecewise_gate_status",
                "orthogonality_status",
                "orthogonality_score",
                "pair_abs_corr_mean",
                "residual_complementarity_status",
                "residual_gain_mean",
                "semantic_dedup_status",
                "semantic_unique_ratio",
                "gate_basis_count",
                "selected_regime_count",
                "basis_count",
                "output_expression_count",
                "regime_structure_json",
                "basis_structure_json",
                "assembler_structure_json",
                "piecewise_gate_basis_json",
                "fold_summary_json",
                "symbolic_family_signature",
                "search_family_signature_contracts_json",
                "fold_count",
                "rmse_mean",
                "rmse_std",
                "rmse_drift",
                "coverage_error_mean",
                "pinaw_mean",
                "interval_score_mean",
                "picp_mean",
                "mean_width_mean",
                "family_concentration",
                "feature_concentration",
                "exact_basis_hit_score",
                "exact_term_recovery_score",
                "outer_objective_score",
                "inner_fit_score",
                "truth_contract_recovery_json",
                "orthogonal_search_objective_json",
                "source_report_json",
                "artifact_record_json",
            )
            artifact_update_columns = tuple(
                name for name in artifact_insert_columns if name not in {"run_id", "artifact_id"}
            )
            artifact_row = {
                "run_id": str(artifact_catalog.get("run_id")),
                "artifact_id": str(artifact_id),
                "trainer_name": artifact_catalog.get("trainer_name"),
                "artifact_kind": artifact_catalog.get("artifact_kind"),
                "head_task": artifact_catalog.get("head_task"),
                "head_semantics_json": (
                    None if artifact_catalog.get("head_semantics_json") is None else _json_dumps(artifact_catalog.get("head_semantics_json"))
                ),
                "complexity_metrics_json": (
                    None
                    if artifact_catalog.get("complexity_metrics_json") is None
                    else _json_dumps(artifact_catalog.get("complexity_metrics_json"))
                ),
                "stability_metrics_json": (
                    None
                    if artifact_catalog.get("stability_metrics_json") is None
                    else _json_dumps(artifact_catalog.get("stability_metrics_json"))
                ),
                "regime_mode": artifact_catalog.get("regime_mode"),
                "basis_scope": artifact_catalog.get("basis_scope"),
                "assembler_mode": artifact_catalog.get("assembler_mode"),
                "piecewise_gate_status": artifact_catalog.get("piecewise_gate_status"),
                "orthogonality_status": artifact_catalog.get("orthogonality_status"),
                "orthogonality_score": artifact_catalog.get("orthogonality_score"),
                "pair_abs_corr_mean": artifact_catalog.get("pair_abs_corr_mean"),
                "residual_complementarity_status": artifact_catalog.get("residual_complementarity_status"),
                "residual_gain_mean": artifact_catalog.get("residual_gain_mean"),
                "semantic_dedup_status": artifact_catalog.get("semantic_dedup_status"),
                "semantic_unique_ratio": artifact_catalog.get("semantic_unique_ratio"),
                "gate_basis_count": artifact_catalog.get("gate_basis_count"),
                "selected_regime_count": artifact_catalog.get("selected_regime_count"),
                "basis_count": artifact_catalog.get("basis_count"),
                "output_expression_count": artifact_catalog.get("output_expression_count"),
                "regime_structure_json": (
                    None if artifact_catalog.get("regime_structure_json") is None else _json_dumps(artifact_catalog.get("regime_structure_json"))
                ),
                "basis_structure_json": (
                    None if artifact_catalog.get("basis_structure_json") is None else _json_dumps(artifact_catalog.get("basis_structure_json"))
                ),
                "assembler_structure_json": (
                    None
                    if artifact_catalog.get("assembler_structure_json") is None
                    else _json_dumps(artifact_catalog.get("assembler_structure_json"))
                ),
                "piecewise_gate_basis_json": (
                    None
                    if artifact_catalog.get("piecewise_gate_basis_json") is None
                    else _json_dumps(artifact_catalog.get("piecewise_gate_basis_json"))
                ),
                "fold_summary_json": (
                    None if artifact_catalog.get("fold_summary_json") is None else _json_dumps(artifact_catalog.get("fold_summary_json"))
                ),
                "symbolic_family_signature": artifact_catalog.get("symbolic_family_signature"),
                "search_family_signature_contracts_json": (
                    None
                    if artifact_catalog.get("search_family_signature_contracts_json") is None
                    else _json_dumps(artifact_catalog.get("search_family_signature_contracts_json"))
                ),
                "fold_count": artifact_catalog.get("fold_count"),
                "rmse_mean": artifact_catalog.get("rmse_mean"),
                "rmse_std": artifact_catalog.get("rmse_std"),
                "rmse_drift": artifact_catalog.get("rmse_drift"),
                "coverage_error_mean": artifact_catalog.get("coverage_error_mean"),
                "pinaw_mean": artifact_catalog.get("pinaw_mean"),
                "interval_score_mean": artifact_catalog.get("interval_score_mean"),
                "picp_mean": artifact_catalog.get("picp_mean"),
                "mean_width_mean": artifact_catalog.get("mean_width_mean"),
                "family_concentration": artifact_catalog.get("family_concentration"),
                "feature_concentration": artifact_catalog.get("feature_concentration"),
                "exact_basis_hit_score": artifact_catalog.get("exact_basis_hit_score"),
                "exact_term_recovery_score": artifact_catalog.get("exact_term_recovery_score"),
                "outer_objective_score": artifact_catalog.get("outer_objective_score"),
                "inner_fit_score": artifact_catalog.get("inner_fit_score"),
                "truth_contract_recovery_json": (
                    None
                    if artifact_catalog.get("truth_contract_recovery_json") is None
                    else _json_dumps(artifact_catalog.get("truth_contract_recovery_json"))
                ),
                "orthogonal_search_objective_json": (
                    None
                    if artifact_catalog.get("orthogonal_search_objective_json") is None
                    else _json_dumps(artifact_catalog.get("orthogonal_search_objective_json"))
                ),
                "source_report_json": (
                    None if artifact_catalog.get("source_report_json") is None else _json_dumps(artifact_catalog.get("source_report_json"))
                ),
                "artifact_record_json": (
                    None if artifact_catalog.get("artifact_record_json") is None else _json_dumps(artifact_catalog.get("artifact_record_json"))
                ),
            }
            conn.execute(
                f"""
                INSERT INTO experiment_artifact_catalog ({", ".join(artifact_insert_columns)})
                VALUES ({", ".join(["?"] * len(artifact_insert_columns))})
                ON CONFLICT(run_id, artifact_id) DO UPDATE SET
                    {", ".join(f"{name}=excluded.{name}" for name in artifact_update_columns)}
                """,
                tuple(artifact_row[name] for name in artifact_insert_columns),
            )

    def _insert_training_trace(self, conn: Any, artifact: Any) -> int:
        if self._run_id is None:
            return 0
        metadata_raw = getattr(artifact, "metadata", {})
        metadata = metadata_raw if isinstance(metadata_raw, Mapping) else {}
        trace_raw = metadata.get("search_trace", {})
        trace = trace_raw if isinstance(trace_raw, Mapping) else {}
        iterations_raw = trace.get("iterations", [])
        if not isinstance(iterations_raw, Sequence) or isinstance(iterations_raw, (str, bytes)):
            return 0

        inserted = 0
        for row in iterations_raw:
            if not isinstance(row, Mapping):
                continue
            iteration = self._as_int_or_none(row.get("iteration"))
            if iteration is None:
                continue

            selected_raw = row.get("selected", {})
            selected = selected_raw if isinstance(selected_raw, Mapping) else {}
            grad_raw = row.get("gradient_summary", {})
            grad = grad_raw if isinstance(grad_raw, Mapping) else {}
            readout_raw = row.get("readout", {})
            readout = readout_raw if isinstance(readout_raw, Mapping) else {}
            readout_before_raw = readout.get("before", {})
            readout_after_raw = readout.get("after", {})
            readout_before = readout_before_raw if isinstance(readout_before_raw, Mapping) else {}
            readout_after = readout_after_raw if isinstance(readout_after_raw, Mapping) else {}
            metrics_before_raw = row.get("metrics_before", {})
            metrics_after_raw = row.get("metrics_after", {})
            metrics_before = metrics_before_raw if isinstance(metrics_before_raw, Mapping) else {}
            metrics_after = metrics_after_raw if isinstance(metrics_after_raw, Mapping) else {}
            metrics_before_val_raw = row.get("metrics_before_val", {})
            metrics_after_val_raw = row.get("metrics_after_val", {})
            metrics_before_val = metrics_before_val_raw if isinstance(metrics_before_val_raw, Mapping) else {}
            metrics_after_val = metrics_after_val_raw if isinstance(metrics_after_val_raw, Mapping) else {}

            operation = selected.get("operation", row.get("stop_reason", None))
            conn.execute(
                """
                INSERT INTO experiment_training_trace (
                    run_id, iteration, ts_utc, operation, selected_name, selected_family, selected_expr,
                    n_terms_before, n_terms_after, rmse_before, rmse_after, val_rmse_before, val_rmse_after,
                    grad_overall_mismatch, weight_l2_before, weight_l2_after, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(self._run_id),
                    int(iteration),
                    _utc_now_iso(),
                    None if operation is None else str(operation),
                    None if selected.get("name") is None else str(selected.get("name")),
                    None if selected.get("family") is None else str(selected.get("family")),
                    None if selected.get("expr") is None else str(selected.get("expr")),
                    self._as_int_or_none(row.get("n_terms_before")),
                    self._as_int_or_none(row.get("n_terms_after")),
                    self._as_float_or_none(metrics_before.get("rmse")),
                    self._as_float_or_none(metrics_after.get("rmse")),
                    self._as_float_or_none(metrics_before_val.get("rmse")),
                    self._as_float_or_none(metrics_after_val.get("rmse")),
                    self._as_float_or_none(grad.get("overall_mismatch")),
                    self._as_float_or_none(readout_before.get("weight_l2")),
                    self._as_float_or_none(readout_after.get("weight_l2")),
                    self._payload_to_json(row),
                ),
            )
            inserted += 1
        return int(inserted)

    def _insert_metrics(self, conn: Any, context: MutableMapping[str, Any]) -> None:
        if self._run_id is None:
            return
        raw = context.get("metrics", {})
        metrics = raw if isinstance(raw, Mapping) else {}
        now = _utc_now_iso()
        for split, row in metrics.items():
            if not isinstance(row, Mapping):
                continue
            for metric, value in row.items():
                try:
                    f = float(value)
                except Exception:
                    continue
                if not math.isfinite(f):
                    continue
                conn.execute(
                    """
                    INSERT INTO experiment_metrics (run_id, ts_utc, split, metric, value)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (str(self._run_id), now, str(split), str(metric), float(f)),
                )

    def on_flow_start(self, context: MutableMapping[str, Any]) -> None:
        self._close_session(commit=True)
        self._event_seq = 0
        self._trace_rows = 0
        self._run_id = self._new_run_id()
        run_name = str(context.get("run_name", "train_flow"))
        model_spec = context.get("model_spec")
        model_spec_json = json.dumps(_to_jsonable(model_spec), ensure_ascii=False)

        def _writer(conn: Any) -> None:
            conn.execute(
                """
                INSERT INTO experiment_runs (
                    run_id, namespace, tag, run_name, started_at_utc, status, model_spec_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(self._run_id),
                    self.namespace,
                    self.tag,
                    run_name,
                    _utc_now_iso(),
                    "running",
                    model_spec_json,
                ),
            )
            self._append_event(conn, event="on_flow_start", context=context)
        self._write(_writer, ensure_schema=True, force_commit=True)
        self._set_report_info(context, status="running")

    def on_data_ready(self, context: MutableMapping[str, Any]) -> None:
        def _writer(conn: Any) -> None:
            self._append_event(conn, event="on_data_ready", context=context)
        self._write(_writer)

    def on_pre_fit(self, context: MutableMapping[str, Any]) -> None:
        def _writer(conn: Any) -> None:
            self._append_event(conn, event="on_pre_fit", context=context)
        self._write(_writer)

    def on_post_fit(self, context: MutableMapping[str, Any]) -> None:
        artifact = context.get("artifact")
        payload = {"artifact_id": getattr(artifact, "artifact_id", None)}
        def _writer(conn: Any) -> None:
            self._trace_rows += int(self._insert_training_trace(conn, artifact))
            self._append_event(conn, event="on_post_fit", context=context, payload=payload)
        self._write(_writer)
        self._set_report_info(context, status="running")

    def on_pre_eval(self, context: MutableMapping[str, Any]) -> None:
        payload = {"eval_splits": context.get("eval_splits")}
        def _writer(conn: Any) -> None:
            self._append_event(conn, event="on_pre_eval", context=context, payload=payload)
        self._write(_writer)

    def on_post_eval(self, context: MutableMapping[str, Any]) -> None:
        def _writer(conn: Any) -> None:
            self._insert_metrics(conn, context)
            self._append_event(conn, event="on_post_eval", context=context, payload={"metrics": context.get("metrics")})
        self._write(_writer)

    def on_pre_persist(self, context: MutableMapping[str, Any]) -> None:
        self._set_report_info(context, status="running")
        def _writer(conn: Any) -> None:
            self._append_event(conn, event="on_pre_persist", context=context)
        self._write(_writer)

    def on_post_persist(self, context: MutableMapping[str, Any]) -> None:
        def _writer(conn: Any) -> None:
            self._append_event(
                conn,
                event="on_post_persist",
                context=context,
                payload={"output_dir": context.get("output_dir")},
            )
        self._write(_writer)

    def on_flow_finish(self, context: MutableMapping[str, Any]) -> None:
        if self._run_id is None:
            return
        report = context.get("report")
        trainer_name = None
        metrics_raw = context.get("metrics")
        metrics_json = json.dumps(_to_jsonable(metrics_raw), ensure_ascii=False)
        report_json = json.dumps(_to_jsonable(report), ensure_ascii=False)
        output_dir = context.get("output_dir")
        trainer = context.get("trainer")
        if trainer is not None:
            trainer_name = str(getattr(trainer, "name", type(trainer).__name__))
        finished_at_utc = _utc_now_iso()

        def _writer(conn: Any) -> None:
            self._append_event(conn, event="on_flow_finish", context=context)
            conn.execute(
                """
                UPDATE experiment_runs
                SET finished_at_utc = ?, status = ?, trainer_name = ?, output_dir = ?, metrics_json = ?, report_json = ?
                WHERE run_id = ?
                """,
                (
                    finished_at_utc,
                    "finished",
                    trainer_name,
                    None if output_dir is None else str(output_dir),
                    metrics_json,
                    report_json,
                    str(self._run_id),
                ),
            )
            self._upsert_catalog_rows(conn, context, status="finished", finished_at_utc=finished_at_utc)
        self._write(_writer, force_commit=True)
        self._close_session(commit=False)
        self._set_report_info(context, status="finished")

    def on_flow_error(self, error: Exception, context: MutableMapping[str, Any]) -> None:
        if self._run_id is None:
            return
        report = context.get("report")
        trainer_name = None
        metrics_raw = context.get("metrics")
        metrics_json = json.dumps(_to_jsonable(metrics_raw), ensure_ascii=False)
        report_json = json.dumps(_to_jsonable(report), ensure_ascii=False)
        output_dir = context.get("output_dir")
        trainer = context.get("trainer")
        if trainer is not None:
            trainer_name = str(getattr(trainer, "name", type(trainer).__name__))
        error_text = f"{type(error).__name__}: {error}"
        finished_at_utc = _utc_now_iso()
        payload = {
            "error": error_text,
            "failed_stage": context.get("failed_stage"),
        }

        def _writer(conn: Any) -> None:
            self._append_event(conn, event="on_flow_error", context=context, payload=payload)
            conn.execute(
                """
                UPDATE experiment_runs
                SET finished_at_utc = ?, status = ?, trainer_name = ?, output_dir = ?, metrics_json = ?, report_json = ?, error_text = ?
                WHERE run_id = ?
                """,
                (
                    finished_at_utc,
                    "failed",
                    trainer_name,
                    None if output_dir is None else str(output_dir),
                    metrics_json,
                    report_json,
                    error_text,
                    str(self._run_id),
                ),
            )
            self._upsert_catalog_rows(conn, context, status="failed", finished_at_utc=finished_at_utc)

        self._write(_writer, force_commit=True)
        self._close_session(commit=False)
        self._set_report_info(context, status="failed")

    def get_context_contract(self) -> Dict[str, Any]:
        return {
            "requires": tuple(self.context_requires),
            "provides": tuple(self.context_provides),
            "mutates": tuple(self.context_mutates),
            "cache": tuple(self.context_cache),
            "notes": self.context_notes,
        }


def build_experiment_tracker_capability(**kwargs: Any) -> ExperimentTrackerCapability:
    params = dict(kwargs)
    return ExperimentTrackerCapability(
        name=str(params.pop("name", "experiment_tracker")),
        db_path=str(params.pop("db_path", resolve_experiment_db_target())),
        namespace=str(params.pop("namespace", "default")),
        tag=params.pop("tag", None),
        report_key=str(params.pop("report_key", "experiment_tracker")),
        max_payload_chars=int(params.pop("max_payload_chars", 25000)),
        io_mode=str(params.pop("io_mode", "batched")),
        commit_interval=int(params.pop("commit_interval", 12)),
        priority=int(params.pop("priority", 0)),
        enabled=bool(params.pop("enabled", True)),
        is_algorithmic=bool(params.pop("is_algorithmic", False)),
        config=dict(params.pop("config", {})),
        context_requires=tuple(str(x) for x in params.pop("context_requires", ("run_name",))),
        context_provides=tuple(str(x) for x in params.pop("context_provides", ("experiment_tracker",))),
        context_mutates=tuple(str(x) for x in params.pop("context_mutates", ("report",))),
        context_cache=tuple(str(x) for x in params.pop("context_cache", tuple())),
        context_notes=params.pop(
            "context_notes",
            "Persists run events and metrics to the resolved experiment DB target for experiment visualization.",
        ),
    )


def _connect_tracker_db(db_path: str | Path) -> Any:
    return open_experiment_db(str(db_path))


def _decode_catalog_row(row: Any, *, json_fields: Sequence[str]) -> dict[str, Any]:
    return decode_row(row, json_fields=json_fields, json_loader=_json_loads)


def list_experiment_run_catalog(
    db_path: str | Path,
    *,
    status: str | None = None,
    trainer_name: str | None = None,
    surface_key: str | None = None,
    family_ref: str | None = None,
    assembly_signature: str | None = None,
    regime_mode: str | None = None,
    basis_scope: str | None = None,
    assembler_mode: str | None = None,
    piecewise_gate_status: str | None = None,
    orthogonality_status: str | None = None,
    residual_complementarity_status: str | None = None,
    semantic_dedup_status: str | None = None,
    has_fold_summary: bool | None = None,
    max_rmse_std: float | None = None,
    max_coverage_error_mean: float | None = None,
    min_exact_basis_hit_score: float | None = None,
    min_exact_term_recovery_score: float | None = None,
    min_outer_objective_score: float | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[Any] = []
    if status is not None:
        where.append("status = ?")
        params.append(str(status))
    if trainer_name is not None:
        where.append("trainer_name = ?")
        params.append(str(trainer_name))
    if surface_key is not None:
        where.append("surface_key = ?")
        params.append(str(surface_key))
    if family_ref is not None:
        where.append("family_ref = ?")
        params.append(str(family_ref))
    if assembly_signature is not None:
        where.append("assembly_signature = ?")
        params.append(str(assembly_signature))
    if regime_mode is not None:
        where.append("regime_mode = ?")
        params.append(str(regime_mode))
    if basis_scope is not None:
        where.append("basis_scope = ?")
        params.append(str(basis_scope))
    if assembler_mode is not None:
        where.append("assembler_mode = ?")
        params.append(str(assembler_mode))
    if piecewise_gate_status is not None:
        where.append("piecewise_gate_status = ?")
        params.append(str(piecewise_gate_status))
    if orthogonality_status is not None:
        where.append("orthogonality_status = ?")
        params.append(str(orthogonality_status))
    if residual_complementarity_status is not None:
        where.append("residual_complementarity_status = ?")
        params.append(str(residual_complementarity_status))
    if semantic_dedup_status is not None:
        where.append("semantic_dedup_status = ?")
        params.append(str(semantic_dedup_status))
    if has_fold_summary is True:
        where.append("fold_summary_json IS NOT NULL")
    elif has_fold_summary is False:
        where.append("fold_summary_json IS NULL")
    if max_rmse_std is not None:
        where.append("rmse_std IS NOT NULL AND rmse_std <= ?")
        params.append(float(max_rmse_std))
    if max_coverage_error_mean is not None:
        where.append("coverage_error_mean IS NOT NULL AND coverage_error_mean <= ?")
        params.append(float(max_coverage_error_mean))
    if min_exact_basis_hit_score is not None:
        where.append("exact_basis_hit_score IS NOT NULL AND exact_basis_hit_score >= ?")
        params.append(float(min_exact_basis_hit_score))
    if min_exact_term_recovery_score is not None:
        where.append("exact_term_recovery_score IS NOT NULL AND exact_term_recovery_score >= ?")
        params.append(float(min_exact_term_recovery_score))
    if min_outer_objective_score is not None:
        where.append("outer_objective_score IS NOT NULL AND outer_objective_score >= ?")
        params.append(float(min_outer_objective_score))

    sql = """
        SELECT *
        FROM experiment_run_catalog
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    params.append(max(1, int(limit)))

    with _connect_tracker_db(db_path) as conn:
        if getattr(conn, "backend", "sqlite") == "postgresql":
            sql += " ORDER BY finished_at_utc DESC NULLS LAST, started_at_utc DESC NULLS LAST, run_id DESC LIMIT ?"
        else:
            sql += " ORDER BY rowid DESC LIMIT ?"
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [
        _decode_catalog_row(
            row,
            json_fields=(
                "task_signature_json",
                "symbolic_family_json",
                "search_mechanism_contracts_json",
                "search_family_signature_contracts_json",
                "search_mechanism_keys_json",
                "search_family_signature_keys_json",
                "compatibility_json",
                "compatibility_drift_json",
                "regime_structure_json",
                "basis_structure_json",
                "assembler_structure_json",
                "piecewise_gate_basis_json",
                "fold_summary_json",
                "truth_contract_recovery_json",
                "orthogonal_search_objective_json",
                "artifact_catalog_json",
                "report_json",
                "surface_record_json",
                "assembly_record_json",
                "run_record_json",
            ),
        )
        for row in rows
    ]


def show_experiment_run_catalog_entry(
    db_path: str | Path,
    *,
    run_id: str,
) -> dict[str, Any] | None:
    with _connect_tracker_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiment_run_catalog WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
    if row is None:
        return None
    return _decode_catalog_row(
        row,
        json_fields=(
            "task_signature_json",
            "symbolic_family_json",
            "search_mechanism_contracts_json",
            "search_family_signature_contracts_json",
            "search_mechanism_keys_json",
            "search_family_signature_keys_json",
            "compatibility_json",
            "compatibility_drift_json",
            "regime_structure_json",
            "basis_structure_json",
            "assembler_structure_json",
            "piecewise_gate_basis_json",
            "fold_summary_json",
            "truth_contract_recovery_json",
            "orthogonal_search_objective_json",
            "artifact_catalog_json",
            "report_json",
            "surface_record_json",
            "assembly_record_json",
            "run_record_json",
        ),
    )


def list_experiment_artifact_catalog(
    db_path: str | Path,
    *,
    trainer_name: str | None = None,
    head_task: str | None = None,
    regime_mode: str | None = None,
    basis_scope: str | None = None,
    assembler_mode: str | None = None,
    piecewise_gate_status: str | None = None,
    orthogonality_status: str | None = None,
    residual_complementarity_status: str | None = None,
    semantic_dedup_status: str | None = None,
    has_fold_summary: bool | None = None,
    max_rmse_std: float | None = None,
    max_coverage_error_mean: float | None = None,
    min_exact_basis_hit_score: float | None = None,
    min_exact_term_recovery_score: float | None = None,
    min_outer_objective_score: float | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    where: list[str] = []
    params: list[Any] = []
    if trainer_name is not None:
        where.append("trainer_name = ?")
        params.append(str(trainer_name))
    if head_task is not None:
        where.append("head_task = ?")
        params.append(str(head_task))
    if regime_mode is not None:
        where.append("regime_mode = ?")
        params.append(str(regime_mode))
    if basis_scope is not None:
        where.append("basis_scope = ?")
        params.append(str(basis_scope))
    if assembler_mode is not None:
        where.append("assembler_mode = ?")
        params.append(str(assembler_mode))
    if piecewise_gate_status is not None:
        where.append("piecewise_gate_status = ?")
        params.append(str(piecewise_gate_status))
    if orthogonality_status is not None:
        where.append("orthogonality_status = ?")
        params.append(str(orthogonality_status))
    if residual_complementarity_status is not None:
        where.append("residual_complementarity_status = ?")
        params.append(str(residual_complementarity_status))
    if semantic_dedup_status is not None:
        where.append("semantic_dedup_status = ?")
        params.append(str(semantic_dedup_status))
    if has_fold_summary is True:
        where.append("fold_summary_json IS NOT NULL")
    elif has_fold_summary is False:
        where.append("fold_summary_json IS NULL")
    if max_rmse_std is not None:
        where.append("rmse_std IS NOT NULL AND rmse_std <= ?")
        params.append(float(max_rmse_std))
    if max_coverage_error_mean is not None:
        where.append("coverage_error_mean IS NOT NULL AND coverage_error_mean <= ?")
        params.append(float(max_coverage_error_mean))
    if min_exact_basis_hit_score is not None:
        where.append("exact_basis_hit_score IS NOT NULL AND exact_basis_hit_score >= ?")
        params.append(float(min_exact_basis_hit_score))
    if min_exact_term_recovery_score is not None:
        where.append("exact_term_recovery_score IS NOT NULL AND exact_term_recovery_score >= ?")
        params.append(float(min_exact_term_recovery_score))
    if min_outer_objective_score is not None:
        where.append("outer_objective_score IS NOT NULL AND outer_objective_score >= ?")
        params.append(float(min_outer_objective_score))

    sql = """
        SELECT *
        FROM experiment_artifact_catalog
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    params.append(max(1, int(limit)))

    with _connect_tracker_db(db_path) as conn:
        if getattr(conn, "backend", "sqlite") == "postgresql":
            sql += " ORDER BY run_id DESC, artifact_id ASC LIMIT ?"
        else:
            sql += " ORDER BY rowid DESC LIMIT ?"
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [
        _decode_catalog_row(
            row,
            json_fields=(
                "head_semantics_json",
                "complexity_metrics_json",
                "stability_metrics_json",
                "regime_structure_json",
                "basis_structure_json",
                "assembler_structure_json",
                "piecewise_gate_basis_json",
                "fold_summary_json",
                "truth_contract_recovery_json",
                "orthogonal_search_objective_json",
                "search_family_signature_contracts_json",
                "source_report_json",
                "artifact_record_json",
            ),
        )
        for row in rows
    ]


def show_experiment_artifact_catalog_entry(
    db_path: str | Path,
    *,
    run_id: str,
    artifact_id: str,
) -> dict[str, Any] | None:
    with _connect_tracker_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiment_artifact_catalog WHERE run_id = ? AND artifact_id = ?",
            (str(run_id), str(artifact_id)),
        ).fetchone()
    if row is None:
        return None
    return _decode_catalog_row(
        row,
        json_fields=(
            "head_semantics_json",
            "complexity_metrics_json",
            "stability_metrics_json",
            "regime_structure_json",
            "basis_structure_json",
            "assembler_structure_json",
            "piecewise_gate_basis_json",
            "fold_summary_json",
            "truth_contract_recovery_json",
            "orthogonal_search_objective_json",
            "search_family_signature_contracts_json",
            "source_report_json",
            "artifact_record_json",
        ),
    )


def experiment_tracker_summary(db_path: str | Path) -> dict[str, Any]:
    target = resolve_experiment_db_target(str(db_path))
    with _connect_tracker_db(target) as conn:
        return {
            "db_path": str(target),
            "backend": str(getattr(conn, "backend", "sqlite")),
            "tables": {
                "experiment_runs": table_count(conn, "experiment_runs"),
                "experiment_events": table_count(conn, "experiment_events"),
                "experiment_metrics": table_count(conn, "experiment_metrics"),
                "experiment_training_trace": table_count(conn, "experiment_training_trace"),
                "experiment_run_catalog": table_count(conn, "experiment_run_catalog"),
                "experiment_artifact_catalog": table_count(conn, "experiment_artifact_catalog"),
            },
        }


def experiment_catalog_table_state(db_path: str | Path) -> dict[str, bool]:
    target = resolve_experiment_db_target(str(db_path))
    with _connect_tracker_db(target) as conn:
        return {
            "run_catalog": table_exists(conn, "experiment_run_catalog"),
            "artifact_catalog": table_exists(conn, "experiment_artifact_catalog"),
            "runs": table_exists(conn, "experiment_runs"),
            "events": table_exists(conn, "experiment_events"),
            "metrics": table_exists(conn, "experiment_metrics"),
            "trace": table_exists(conn, "experiment_training_trace"),
        }


def experiment_catalog_filter_values(db_path: str | Path) -> dict[str, list[str]]:
    target = resolve_experiment_db_target(str(db_path))
    out = {
        "run_status": [],
        "run_trainer_name": [],
        "run_surface_key": [],
        "run_family_ref": [],
        "run_regime_mode": [],
        "run_basis_scope": [],
        "run_assembler_mode": [],
        "run_piecewise_gate_status": [],
        "run_orthogonality_status": [],
        "run_residual_complementarity_status": [],
        "run_semantic_dedup_status": [],
        "artifact_trainer_name": [],
        "artifact_head_task": [],
        "artifact_regime_mode": [],
        "artifact_basis_scope": [],
        "artifact_assembler_mode": [],
        "artifact_piecewise_gate_status": [],
        "artifact_orthogonality_status": [],
        "artifact_residual_complementarity_status": [],
        "artifact_semantic_dedup_status": [],
    }
    with _connect_tracker_db(target) as conn:
        try:
            if table_exists(conn, "experiment_run_catalog"):
                out["run_status"] = first_column_texts(conn.execute("SELECT DISTINCT status FROM experiment_run_catalog WHERE status IS NOT NULL AND TRIM(status) <> '' ORDER BY status ASC").fetchall())
                out["run_trainer_name"] = first_column_texts(conn.execute("SELECT DISTINCT trainer_name FROM experiment_run_catalog WHERE trainer_name IS NOT NULL AND TRIM(trainer_name) <> '' ORDER BY trainer_name ASC").fetchall())
                out["run_surface_key"] = first_column_texts(conn.execute("SELECT DISTINCT surface_key FROM experiment_run_catalog WHERE surface_key IS NOT NULL AND TRIM(surface_key) <> '' ORDER BY surface_key ASC").fetchall())
                out["run_family_ref"] = first_column_texts(conn.execute("SELECT DISTINCT family_ref FROM experiment_run_catalog WHERE family_ref IS NOT NULL AND TRIM(family_ref) <> '' ORDER BY family_ref ASC").fetchall())
                out["run_regime_mode"] = first_column_texts(conn.execute("SELECT DISTINCT regime_mode FROM experiment_run_catalog WHERE regime_mode IS NOT NULL AND TRIM(regime_mode) <> '' ORDER BY regime_mode ASC").fetchall())
                out["run_basis_scope"] = first_column_texts(conn.execute("SELECT DISTINCT basis_scope FROM experiment_run_catalog WHERE basis_scope IS NOT NULL AND TRIM(basis_scope) <> '' ORDER BY basis_scope ASC").fetchall())
                out["run_assembler_mode"] = first_column_texts(conn.execute("SELECT DISTINCT assembler_mode FROM experiment_run_catalog WHERE assembler_mode IS NOT NULL AND TRIM(assembler_mode) <> '' ORDER BY assembler_mode ASC").fetchall())
                out["run_piecewise_gate_status"] = first_column_texts(conn.execute("SELECT DISTINCT piecewise_gate_status FROM experiment_run_catalog WHERE piecewise_gate_status IS NOT NULL AND TRIM(piecewise_gate_status) <> '' ORDER BY piecewise_gate_status ASC").fetchall())
                out["run_orthogonality_status"] = first_column_texts(conn.execute("SELECT DISTINCT orthogonality_status FROM experiment_run_catalog WHERE orthogonality_status IS NOT NULL AND TRIM(orthogonality_status) <> '' ORDER BY orthogonality_status ASC").fetchall())
                out["run_residual_complementarity_status"] = first_column_texts(conn.execute("SELECT DISTINCT residual_complementarity_status FROM experiment_run_catalog WHERE residual_complementarity_status IS NOT NULL AND TRIM(residual_complementarity_status) <> '' ORDER BY residual_complementarity_status ASC").fetchall())
                out["run_semantic_dedup_status"] = first_column_texts(conn.execute("SELECT DISTINCT semantic_dedup_status FROM experiment_run_catalog WHERE semantic_dedup_status IS NOT NULL AND TRIM(semantic_dedup_status) <> '' ORDER BY semantic_dedup_status ASC").fetchall())
            if table_exists(conn, "experiment_artifact_catalog"):
                out["artifact_trainer_name"] = first_column_texts(conn.execute("SELECT DISTINCT trainer_name FROM experiment_artifact_catalog WHERE trainer_name IS NOT NULL AND TRIM(trainer_name) <> '' ORDER BY trainer_name ASC").fetchall())
                out["artifact_head_task"] = first_column_texts(conn.execute("SELECT DISTINCT head_task FROM experiment_artifact_catalog WHERE head_task IS NOT NULL AND TRIM(head_task) <> '' ORDER BY head_task ASC").fetchall())
                out["artifact_regime_mode"] = first_column_texts(conn.execute("SELECT DISTINCT regime_mode FROM experiment_artifact_catalog WHERE regime_mode IS NOT NULL AND TRIM(regime_mode) <> '' ORDER BY regime_mode ASC").fetchall())
                out["artifact_basis_scope"] = first_column_texts(conn.execute("SELECT DISTINCT basis_scope FROM experiment_artifact_catalog WHERE basis_scope IS NOT NULL AND TRIM(basis_scope) <> '' ORDER BY basis_scope ASC").fetchall())
                out["artifact_assembler_mode"] = first_column_texts(conn.execute("SELECT DISTINCT assembler_mode FROM experiment_artifact_catalog WHERE assembler_mode IS NOT NULL AND TRIM(assembler_mode) <> '' ORDER BY assembler_mode ASC").fetchall())
                out["artifact_piecewise_gate_status"] = first_column_texts(conn.execute("SELECT DISTINCT piecewise_gate_status FROM experiment_artifact_catalog WHERE piecewise_gate_status IS NOT NULL AND TRIM(piecewise_gate_status) <> '' ORDER BY piecewise_gate_status ASC").fetchall())
                out["artifact_orthogonality_status"] = first_column_texts(conn.execute("SELECT DISTINCT orthogonality_status FROM experiment_artifact_catalog WHERE orthogonality_status IS NOT NULL AND TRIM(orthogonality_status) <> '' ORDER BY orthogonality_status ASC").fetchall())
                out["artifact_residual_complementarity_status"] = first_column_texts(conn.execute("SELECT DISTINCT residual_complementarity_status FROM experiment_artifact_catalog WHERE residual_complementarity_status IS NOT NULL AND TRIM(residual_complementarity_status) <> '' ORDER BY residual_complementarity_status ASC").fetchall())
                out["artifact_semantic_dedup_status"] = first_column_texts(conn.execute("SELECT DISTINCT semantic_dedup_status FROM experiment_artifact_catalog WHERE semantic_dedup_status IS NOT NULL AND TRIM(semantic_dedup_status) <> '' ORDER BY semantic_dedup_status ASC").fetchall())
        except Exception:
            return out
    return out


__all__ = [
    "ExperimentTrackerCapability",
    "build_experiment_tracker_capability",
    "experiment_catalog_filter_values",
    "experiment_catalog_table_state",
    "experiment_tracker_summary",
    "list_experiment_artifact_catalog",
    "list_experiment_run_catalog",
    "show_experiment_artifact_catalog_entry",
    "show_experiment_run_catalog_entry",
]

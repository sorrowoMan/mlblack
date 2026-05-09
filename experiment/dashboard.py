from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.experiment_db import experiment_db_config_info, normalize_experiment_db_target, open_experiment_db, resolve_experiment_db_target, table_exists
from core.flow_experiment_tracker import (
    experiment_catalog_filter_values,
    experiment_catalog_table_state,
    list_experiment_artifact_catalog,
    list_experiment_run_catalog,
    experiment_tracker_summary,
    show_experiment_artifact_catalog_entry,
    show_experiment_run_catalog_entry,
)

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for dashboard. Install with: python -m pip install streamlit"
    ) from exc


def dashboard_script_path() -> Path:
    return Path(__file__).resolve()


def build_streamlit_command(
    *,
    db_path: str,
    limit: int = 500,
    host: str | None = None,
    port: int | None = None,
    headless: bool = False,
) -> list[str]:
    command = [sys.executable, "-m", "streamlit", "run", str(dashboard_script_path())]
    if host:
        command.extend(["--server.address", str(host)])
    if port is not None:
        command.extend(["--server.port", str(int(port))])
    if headless:
        command.extend(["--server.headless", "true"])
    command.extend(["--", "--db", str(db_path), "--limit", str(int(limit))])
    return command


_QUERY_BASE_KEYS: tuple[str, ...] = (
    "db",
    "limit",
    "view",
    "selected",
)


def _normalize_filter_values(value: object) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, (list, tuple, set, frozenset)):
        out: list[str] = []
        for item in value:
            out.extend(_normalize_filter_values(item))
        return tuple(out)
    text = str(value).strip()
    return (text,) if text else ()


def _read_query_params(st_module: Any) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    def _coerce(raw: dict[str, object]) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
        base: dict[str, str] = {}
        filters: dict[str, tuple[str, ...]] = {}
        for key, raw_value in raw.items():
            value = raw_value[-1] if isinstance(raw_value, list) else raw_value
            text = str(value or "").strip()
            if not text:
                continue
            if key in _QUERY_BASE_KEYS:
                base[str(key)] = text
                continue
            if str(key).startswith("f_"):
                field_name = str(key)[2:].strip()
                values = _normalize_filter_values(text)
                if field_name and values:
                    filters[field_name] = values
        return base, filters

    try:
        params = st_module.query_params
        return _coerce({str(key): params.get(key) for key in list(params.keys())})
    except Exception:
        try:
            return _coerce(st_module.experimental_get_query_params())
        except Exception:
            return {}, {}


def _build_query_param_payload(
    *,
    base_params: dict[str, object],
    field_filters: dict[str, object] | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for key, raw_value in dict(base_params).items():
        text = str(raw_value or "").strip()
        if text:
            payload[str(key)] = text
    for field_name, raw_value in dict(field_filters or {}).items():
        values = _normalize_filter_values(raw_value)
        if values:
            payload[f"f_{str(field_name).strip()}"] = ",".join(values)
    return payload


def _build_deep_link_query(
    *,
    base_params: dict[str, object],
    field_filters: dict[str, object] | None = None,
) -> str:
    payload = _build_query_param_payload(base_params=base_params, field_filters=field_filters)
    return "?" + urlencode(payload)


def _write_query_params(
    st_module: Any,
    *,
    base_params: dict[str, object],
    field_filters: dict[str, object] | None = None,
) -> None:
    payload = _build_query_param_payload(base_params=base_params, field_filters=field_filters)
    try:
        params = st_module.query_params
        params.clear()
        for key, value in payload.items():
            params[str(key)] = str(value)
        return
    except Exception:
        pass
    try:
        st_module.experimental_set_query_params(**payload)
    except Exception:
        pass


def _selection_run_key(run_id: str) -> str:
    return f"run:{str(run_id).strip()}"


def _selection_artifact_key(run_id: str, artifact_id: str) -> str:
    return f"artifact:{str(run_id).strip()}:{str(artifact_id).strip()}"


def _decode_selection_key(value: str | None) -> dict[str, str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.startswith("run:"):
        run_id = text.split(":", 1)[1].strip()
        return {"kind": "run", "run_id": run_id} if run_id else None
    if text.startswith("artifact:"):
        _, rest = text.split(":", 1)
        parts = rest.split(":", 1)
        if len(parts) != 2:
            return None
        run_id = parts[0].strip()
        artifact_id = parts[1].strip()
        if run_id and artifact_id:
            return {"kind": "artifact", "run_id": run_id, "artifact_id": artifact_id}
    return None


def _query_filter_first(query_filters: dict[str, tuple[str, ...]], name: str, default: str = "") -> str:
    values = query_filters.get(str(name), ())
    return str(values[0]).strip() if values else str(default).strip()


def _fold_choice_from_query(value: str | None) -> str:
    mapping = {
        "present": "有 / Present",
        "absent": "无 / Absent",
        "any": "不限 / Any",
    }
    return mapping.get(str(value or "").strip().lower(), "不限 / Any")


def _fold_query_token(choice: str | None) -> str:
    mapping = {
        "有 / Present": "present",
        "无 / Absent": "absent",
        "不限 / Any": "any",
    }
    return mapping.get(str(choice or "").strip(), "any")


def _apply_threshold_query_state(
    session_state: Any,
    *,
    key_prefix: str,
    raw_value: str | None,
    default_value: float,
) -> None:
    enabled_key = f"{key_prefix}_enabled"
    value_key = f"{key_prefix}_value"
    text = str(raw_value or "").strip()
    if not text:
        session_state.setdefault(enabled_key, False)
        session_state.setdefault(value_key, float(default_value))
        return
    try:
        numeric = float(text)
    except Exception:
        session_state.setdefault(enabled_key, False)
        session_state.setdefault(value_key, float(default_value))
        return
    session_state.setdefault(enabled_key, True)
    session_state.setdefault(value_key, float(numeric))


def _query_frame(conn: Any, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    rows = conn.execute(sql, tuple(params)).fetchall()
    materialized: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            materialized.append({str(key): value for key, value in dict(row).items()})
        else:
            materialized.append({str(key): row[key] for key in row.keys()})
    return pd.DataFrame(materialized)


def _load_runs(conn: Any, *, limit: int) -> pd.DataFrame:
    try:
        return _query_frame(
            conn,
            """
            SELECT
                r.run_id,
                r.namespace,
                r.tag,
                r.run_name,
                r.status,
                r.trainer_name,
                r.started_at_utc,
                r.finished_at_utc,
                r.output_dir,
                COALESCE(t.trace_rows, 0) AS trace_rows
            FROM experiment_runs r
            LEFT JOIN (
                SELECT run_id, COUNT(*) AS trace_rows
                FROM experiment_training_trace
                GROUP BY run_id
            ) t
            ON t.run_id = r.run_id
            ORDER BY r.started_at_utc DESC
            LIMIT ?
            """,
            (int(limit),),
        )
    except Exception:
        return _query_frame(
            conn,
            """
            SELECT run_id, namespace, tag, run_name, status, trainer_name, started_at_utc, finished_at_utc, output_dir
            FROM experiment_runs
            ORDER BY started_at_utc DESC
            LIMIT ?
            """,
            (int(limit),),
        )


def _load_events(conn: Any, *, run_id: str) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT seq, ts_utc, event, stage, payload_json
        FROM experiment_events
        WHERE run_id = ?
        ORDER BY seq ASC
        """,
        (str(run_id),),
    )


def _load_metrics(conn: Any, *, run_id: str) -> pd.DataFrame:
    return _query_frame(
        conn,
        """
        SELECT ts_utc, split, metric, value
        FROM experiment_metrics
        WHERE run_id = ?
        ORDER BY ts_utc ASC
        """,
        (str(run_id),),
    )


def _load_training_trace(conn: Any, *, run_id: str) -> pd.DataFrame:
    try:
        return _query_frame(
            conn,
            """
            SELECT
                iteration,
                ts_utc,
                operation,
                selected_name,
                selected_family,
                selected_expr,
                n_terms_before,
                n_terms_after,
                rmse_before,
                rmse_after,
                val_rmse_before,
                val_rmse_after,
                grad_overall_mismatch,
                weight_l2_before,
                weight_l2_after,
                payload_json
            FROM experiment_training_trace
            WHERE run_id = ?
            ORDER BY iteration ASC
            """,
            (str(run_id),),
        )
    except Exception:
        return pd.DataFrame(
            columns=[
                "iteration",
                "ts_utc",
                "operation",
                "selected_name",
                "selected_family",
                "selected_expr",
                "n_terms_before",
                "n_terms_after",
                "rmse_before",
                "rmse_after",
                "val_rmse_before",
                "val_rmse_after",
                "grad_overall_mismatch",
                "weight_l2_before",
                "weight_l2_after",
                "payload_json",
            ]
        )


def _to_chart_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    chart = metrics.copy()
    chart["ts_utc"] = pd.to_datetime(chart["ts_utc"], errors="coerce")
    chart = chart.dropna(subset=["ts_utc"])
    if chart.empty:
        return chart

    pivot = chart.pivot_table(
        index="ts_utc",
        columns=["split", "metric"],
        values="value",
        aggfunc="mean",
    ).sort_index()

    if isinstance(pivot.columns, pd.MultiIndex):
        pivot.columns = [f"{str(a)}.{str(b)}" for a, b in pivot.columns.to_flat_index()]
    else:
        pivot.columns = [str(c) for c in list(pivot.columns)]

    return pivot


def _to_trace_curve_frame(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return trace
    t = trace.copy()
    t = t.sort_values(by=["iteration"], ascending=True)
    cols = [
        c
        for c in (
            "rmse_before",
            "rmse_after",
            "val_rmse_before",
            "val_rmse_after",
            "grad_overall_mismatch",
            "weight_l2_before",
            "weight_l2_after",
        )
        if c in t.columns
    ]
    if not cols:
        return pd.DataFrame()
    out = t.set_index("iteration")[cols]
    return out


def _to_trace_derived_frame(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return trace

    t = trace.copy().sort_values(by=["iteration"], ascending=True)
    for c in (
        "weight_l2_before",
        "weight_l2_after",
        "rmse_before",
        "rmse_after",
        "val_rmse_before",
        "val_rmse_after",
        "grad_overall_mismatch",
    ):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")

    t["delta_weight_l2"] = t["weight_l2_after"] - t["weight_l2_before"]
    t["delta_train_rmse"] = t["rmse_before"] - t["rmse_after"]
    t["delta_val_rmse"] = t["val_rmse_before"] - t["val_rmse_after"]
    t["delta_effective_rmse"] = t["delta_val_rmse"].where(t["delta_val_rmse"].notna(), t["delta_train_rmse"])

    grad = t["grad_overall_mismatch"] if "grad_overall_mismatch" in t.columns else pd.Series(dtype=float)
    t["delta_grad_mismatch"] = grad.diff()
    eps = 1e-8
    t["hard_fit_index"] = t["delta_weight_l2"].abs() / (t["delta_grad_mismatch"].abs() + eps)
    return t


def _summarize_replace_attribution(derived: pd.DataFrame) -> pd.DataFrame:
    if derived.empty or "operation" not in derived.columns:
        return pd.DataFrame(columns=["family", "count", "mean_gain", "median_gain", "win_rate"])

    rep = derived[derived["operation"].astype(str).str.lower() == "replace"].copy()
    if rep.empty:
        return pd.DataFrame(columns=["family", "count", "mean_gain", "median_gain", "win_rate"])

    rep["family"] = rep["selected_family"].fillna("N/A").astype(str)
    rep["gain"] = pd.to_numeric(rep["delta_effective_rmse"], errors="coerce")
    rep = rep[rep["gain"].notna()]
    if rep.empty:
        return pd.DataFrame(columns=["family", "count", "mean_gain", "median_gain", "win_rate"])

    rows = (
        rep.groupby("family", as_index=False)
        .agg(
            count=("gain", "size"),
            mean_gain=("gain", "mean"),
            median_gain=("gain", "median"),
            win_rate=("gain", lambda s: float((s > 0).mean())),
        )
        .sort_values(by=["mean_gain", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return rows


def _summarize_operation_transition(derived: pd.DataFrame) -> pd.DataFrame:
    if derived.empty or "operation" not in derived.columns:
        return pd.DataFrame(columns=["transition", "count", "mean_gain", "win_rate"])

    t = derived.copy().sort_values(by=["iteration"], ascending=True)
    t["op"] = t["operation"].fillna("N/A").astype(str)
    t["prev_op"] = t["op"].shift(1)
    t = t[t["prev_op"].notna()].copy()
    if t.empty:
        return pd.DataFrame(columns=["transition", "count", "mean_gain", "win_rate"])

    t["transition"] = t["prev_op"].astype(str) + " -> " + t["op"].astype(str)
    t["gain"] = pd.to_numeric(t["delta_effective_rmse"], errors="coerce")
    t = t[t["gain"].notna()]
    if t.empty:
        return pd.DataFrame(columns=["transition", "count", "mean_gain", "win_rate"])

    rows = (
        t.groupby("transition", as_index=False)
        .agg(
            count=("gain", "size"),
            mean_gain=("gain", "mean"),
            win_rate=("gain", lambda s: float((s > 0).mean())),
        )
        .sort_values(by=["mean_gain", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return rows


def _json_dict(raw: object) -> dict[str, object]:
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _term_identity(*, expr_key: object | None = None, expr: object | None = None, name: object | None = None) -> str:
    for v in (expr_key, expr, name):
        if v is None:
            continue
        text = str(v).strip()
        if text:
            return text
    return ""


def _drop_active_term(active: dict[str, dict[str, object]], *, expr_key: object | None, expr: object | None, name: object | None) -> None:
    key = _term_identity(expr_key=expr_key, expr=expr, name=name)
    if not key:
        return
    if key in active:
        active.pop(key, None)
        return
    for k, item in list(active.items()):
        if str(item.get("expr", "")).strip() == str(expr or "").strip():
            active.pop(k, None)
            return
        if str(item.get("name", "")).strip() == str(name or "").strip():
            active.pop(k, None)
            return


def _replay_function_cluster(trace_like: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trace_like.empty:
        empty_active = pd.DataFrame(columns=["name", "family", "expr", "last_iteration", "last_operation"])
        empty_path = pd.DataFrame(
            columns=["iteration", "operation", "selected_name", "selected_family", "n_terms_estimated", "n_terms_before", "n_terms_after"]
        )
        empty_top = pd.DataFrame(columns=["name", "expr", "coeff_l2", "coeff_max_abs", "coeff_target"])
        return empty_active, empty_path, empty_top

    t = trace_like.copy().sort_values(by=["iteration"], ascending=True)
    active: dict[str, dict[str, object]] = {}
    evolution_rows: list[dict[str, object]] = []
    last_readout_top_terms: list[dict[str, object]] = []

    for _, row in t.iterrows():
        payload = _json_dict(row.get("payload_json"))
        selected_raw = payload.get("selected", {})
        selected = selected_raw if isinstance(selected_raw, dict) else {}
        pruning_raw = payload.get("pruning", {})
        pruning = pruning_raw if isinstance(pruning_raw, dict) else {}

        operation = str(selected.get("operation", row.get("operation", ""))).strip().lower()
        selected_name = str(selected.get("name", row.get("selected_name", ""))).strip()
        selected_family = str(selected.get("family", row.get("selected_family", ""))).strip()
        selected_expr = str(selected.get("expr", row.get("selected_expr", ""))).strip()

        if operation == "replace":
            drop_raw = selected.get("replace_drop", {})
            drop = drop_raw if isinstance(drop_raw, dict) else {}
            _drop_active_term(
                active,
                expr_key=drop.get("expr_key"),
                expr=drop.get("expr"),
                name=drop.get("name"),
            )
        if operation in {"add", "replace"}:
            key = _term_identity(expr=selected_expr, name=selected_name)
            if key:
                active[key] = {
                    "name": selected_name or key,
                    "family": selected_family or "N/A",
                    "expr": selected_expr or key,
                    "last_iteration": int(row.get("iteration", 0)),
                    "last_operation": operation,
                }

        removed_raw = pruning.get("removed_terms", [])
        removed_terms = removed_raw if isinstance(removed_raw, list) else []
        for item in removed_terms:
            if not isinstance(item, dict):
                continue
            _drop_active_term(
                active,
                expr_key=item.get("expr_key"),
                expr=item.get("expr"),
                name=item.get("name"),
            )

        readout_raw = payload.get("readout", {})
        readout = readout_raw if isinstance(readout_raw, dict) else {}
        after_raw = readout.get("after", {})
        after = after_raw if isinstance(after_raw, dict) else {}
        top_terms_raw = after.get("top_terms", [])
        top_terms = top_terms_raw if isinstance(top_terms_raw, list) else []
        last_readout_top_terms = [dict(v) for v in top_terms if isinstance(v, dict)]

        evolution_rows.append(
            {
                "iteration": int(row.get("iteration", 0)),
                "operation": operation or "N/A",
                "selected_name": selected_name or "N/A",
                "selected_family": selected_family or "N/A",
                "n_terms_estimated": int(len(active)),
                "n_terms_before": row.get("n_terms_before", None),
                "n_terms_after": row.get("n_terms_after", None),
                "delta_effective_rmse": row.get("delta_effective_rmse", None),
            }
        )

    active_rows = sorted(
        list(active.values()),
        key=lambda x: (str(x.get("family", "")), str(x.get("name", ""))),
    )
    active_df = pd.DataFrame(active_rows, columns=["name", "family", "expr", "last_iteration", "last_operation"])
    evolution_df = pd.DataFrame(
        evolution_rows,
        columns=[
            "iteration",
            "operation",
            "selected_name",
            "selected_family",
            "n_terms_estimated",
            "n_terms_before",
            "n_terms_after",
            "delta_effective_rmse",
        ],
    )
    top_terms_df = pd.DataFrame(
        last_readout_top_terms,
        columns=["name", "expr", "coeff_l2", "coeff_max_abs", "coeff_target"],
    )
    return active_df, evolution_df, top_terms_df


def _formula_lines_from_readout_top_terms(
    readout_top_terms: pd.DataFrame,
    *,
    max_terms_per_target: int = 10,
    coef_eps: float = 1e-12,
) -> list[str]:
    if readout_top_terms.empty:
        return []

    rows = readout_top_terms.copy()
    if "coeff_l2" in rows.columns:
        rows["coeff_l2"] = pd.to_numeric(rows["coeff_l2"], errors="coerce")
        rows = rows.sort_values(by=["coeff_l2"], ascending=False)

    target_dim = 1
    coeff_vectors: list[list[float]] = []
    for _, row in rows.iterrows():
        raw = row.get("coeff_target", [])
        if isinstance(raw, (list, tuple)):
            vec = [float(v) for v in raw]
        else:
            vec = []
        coeff_vectors.append(vec)
        if len(vec) > target_dim:
            target_dim = int(len(vec))

    formulas: list[str] = []
    for t_idx in range(int(target_dim)):
        pieces: list[str] = []
        used = 0
        for ridx, (_, row) in enumerate(rows.iterrows()):
            expr = str(row.get("expr", "")).strip()
            if not expr:
                continue
            vec = coeff_vectors[ridx] if ridx < len(coeff_vectors) else []
            coef = float(vec[t_idx]) if t_idx < len(vec) else 0.0
            if abs(coef) <= float(coef_eps):
                continue
            pieces.append(f"{coef:.6g}*({expr})")
            used += 1
            if used >= int(max(1, max_terms_per_target)):
                break
        if not pieces:
            continue
        expr_text = " + ".join(pieces).replace("+ -", "- ")
        formulas.append(f"y{t_idx}(x) ~= {expr_text} + b{t_idx}")
    return formulas


def _connect_experiment_db(db_path: str | Path) -> Any:
    return open_experiment_db(str(db_path))


@st.cache_data(show_spinner=False)
def _catalog_table_state(db_path: str, refresh_nonce: int) -> dict[str, bool]:
    del refresh_nonce
    return experiment_catalog_table_state(db_path)


@st.cache_data(show_spinner=False)
def _load_catalog_filter_values(db_path: str, refresh_nonce: int) -> dict[str, list[str]]:
    del refresh_nonce
    return experiment_catalog_filter_values(db_path)


@st.cache_data(show_spinner=False)
def _load_runs_cached(db_path: str, limit: int, refresh_nonce: int) -> pd.DataFrame:
    del refresh_nonce
    with _connect_experiment_db(db_path) as conn:
        return _load_runs(conn, limit=limit)


@st.cache_data(show_spinner=False)
def _load_run_overview_cached(db_path: str, run_id: str, refresh_nonce: int) -> pd.DataFrame:
    del refresh_nonce
    with _connect_experiment_db(db_path) as conn:
        try:
            return _query_frame(
                conn,
                """
                SELECT
                    r.run_id,
                    r.namespace,
                    r.tag,
                    r.run_name,
                    r.status,
                    r.trainer_name,
                    r.started_at_utc,
                    r.finished_at_utc,
                    r.output_dir,
                    COALESCE(t.trace_rows, 0) AS trace_rows
                FROM experiment_runs r
                LEFT JOIN (
                    SELECT run_id, COUNT(*) AS trace_rows
                    FROM experiment_training_trace
                    GROUP BY run_id
                ) t
                ON t.run_id = r.run_id
                WHERE r.run_id = ?
                LIMIT 1
                """,
                (str(run_id),),
            )
        except Exception:
            return _query_frame(
                conn,
                """
                SELECT run_id, namespace, tag, run_name, status, trainer_name, started_at_utc, finished_at_utc, output_dir
                FROM experiment_runs
                WHERE run_id = ?
                LIMIT 1
                """,
                (str(run_id),),
            )


@st.cache_data(show_spinner=False)
def _load_events_cached(db_path: str, run_id: str, refresh_nonce: int) -> pd.DataFrame:
    del refresh_nonce
    with _connect_experiment_db(db_path) as conn:
        return _load_events(conn, run_id=run_id)


@st.cache_data(show_spinner=False)
def _load_metrics_cached(db_path: str, run_id: str, refresh_nonce: int) -> pd.DataFrame:
    del refresh_nonce
    with _connect_experiment_db(db_path) as conn:
        return _load_metrics(conn, run_id=run_id)


@st.cache_data(show_spinner=False)
def _load_training_trace_cached(db_path: str, run_id: str, refresh_nonce: int) -> pd.DataFrame:
    del refresh_nonce
    with _connect_experiment_db(db_path) as conn:
        return _load_training_trace(conn, run_id=run_id)


@st.cache_data(show_spinner=False)
def _load_run_catalog_cached(
    db_path: str,
    status: str | None,
    trainer_name: str | None,
    surface_key: str | None,
    family_ref: str | None,
    assembly_signature: str | None,
    regime_mode: str | None,
    basis_scope: str | None,
    assembler_mode: str | None,
    piecewise_gate_status: str | None,
    orthogonality_status: str | None,
    residual_complementarity_status: str | None,
    semantic_dedup_status: str | None,
    has_fold_summary: bool | None,
    max_rmse_std: float | None,
    max_coverage_error_mean: float | None,
    min_exact_basis_hit_score: float | None,
    min_exact_term_recovery_score: float | None,
    min_outer_objective_score: float | None,
    limit: int,
    refresh_nonce: int,
) -> list[dict[str, Any]]:
    del refresh_nonce
    return list_experiment_run_catalog(
        db_path,
        status=status,
        trainer_name=trainer_name,
        surface_key=surface_key,
        family_ref=family_ref,
        assembly_signature=assembly_signature,
        regime_mode=regime_mode,
        basis_scope=basis_scope,
        assembler_mode=assembler_mode,
        piecewise_gate_status=piecewise_gate_status,
        orthogonality_status=orthogonality_status,
        residual_complementarity_status=residual_complementarity_status,
        semantic_dedup_status=semantic_dedup_status,
        has_fold_summary=has_fold_summary,
        max_rmse_std=max_rmse_std,
        max_coverage_error_mean=max_coverage_error_mean,
        min_exact_basis_hit_score=min_exact_basis_hit_score,
        min_exact_term_recovery_score=min_exact_term_recovery_score,
        min_outer_objective_score=min_outer_objective_score,
        limit=limit,
    )


@st.cache_data(show_spinner=False)
def _load_artifact_catalog_cached(
    db_path: str,
    trainer_name: str | None,
    head_task: str | None,
    regime_mode: str | None,
    basis_scope: str | None,
    assembler_mode: str | None,
    piecewise_gate_status: str | None,
    orthogonality_status: str | None,
    residual_complementarity_status: str | None,
    semantic_dedup_status: str | None,
    has_fold_summary: bool | None,
    max_rmse_std: float | None,
    max_coverage_error_mean: float | None,
    min_exact_basis_hit_score: float | None,
    min_exact_term_recovery_score: float | None,
    min_outer_objective_score: float | None,
    limit: int,
    refresh_nonce: int,
) -> list[dict[str, Any]]:
    del refresh_nonce
    return list_experiment_artifact_catalog(
        db_path,
        trainer_name=trainer_name,
        head_task=head_task,
        regime_mode=regime_mode,
        basis_scope=basis_scope,
        assembler_mode=assembler_mode,
        piecewise_gate_status=piecewise_gate_status,
        orthogonality_status=orthogonality_status,
        residual_complementarity_status=residual_complementarity_status,
        semantic_dedup_status=semantic_dedup_status,
        has_fold_summary=has_fold_summary,
        max_rmse_std=max_rmse_std,
        max_coverage_error_mean=max_coverage_error_mean,
        min_exact_basis_hit_score=min_exact_basis_hit_score,
        min_exact_term_recovery_score=min_exact_term_recovery_score,
        min_outer_objective_score=min_outer_objective_score,
        limit=limit,
    )


@st.cache_data(show_spinner=False)
def _show_run_catalog_cached(db_path: str, run_id: str, refresh_nonce: int) -> dict[str, Any] | None:
    del refresh_nonce
    return show_experiment_run_catalog_entry(db_path, run_id=run_id)


@st.cache_data(show_spinner=False)
def _show_artifact_catalog_cached(
    db_path: str,
    run_id: str,
    artifact_id: str,
    refresh_nonce: int,
) -> dict[str, Any] | None:
    del refresh_nonce
    return show_experiment_artifact_catalog_entry(db_path, run_id=run_id, artifact_id=artifact_id)


def _optional_exact_filter(label: str, options: list[str], *, key: str) -> str | None:
    choice = st.selectbox(label, ["不限 / Any", *options], index=0, key=key)
    return None if choice == "不限 / Any" else str(choice)


def _fold_filter_value(choice: str) -> bool | None:
    if choice == "有 / Present":
        return True
    if choice == "无 / Absent":
        return False
    return None


def _metric_threshold_input(label: str, *, key_prefix: str, default_value: float) -> float | None:
    enabled = bool(
        st.checkbox(
            f"启用 {label} 上限 / Enable {label} ceiling",
            value=False,
            key=f"{key_prefix}_enabled",
        )
    )
    value = float(
        st.number_input(
            f"{label} <= ",
            min_value=0.0,
            value=float(default_value),
            step=0.01,
            format="%.4f",
            key=f"{key_prefix}_value",
            disabled=not enabled,
        )
    )
    return value if enabled else None


def _metric_floor_input(label: str, *, key_prefix: str, default_value: float) -> float | None:
    enabled = bool(
        st.checkbox(
            f"启用 {label} 下限 / Enable {label} floor",
            value=False,
            key=f"{key_prefix}_enabled",
        )
    )
    value = float(
        st.number_input(
            f"{label} >= ",
            min_value=0.0,
            value=float(default_value),
            step=0.01,
            format="%.4f",
            key=f"{key_prefix}_value",
            disabled=not enabled,
        )
    )
    return value if enabled else None


def _metric_text(value: object, *, digits: int = 4) -> str:
    try:
        if value is None:
            return "-"
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return "-"


def _count_fold_summary_rows(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if isinstance(row.get("fold_summary_json"), dict) and bool(dict(row.get("fold_summary_json", {}))))


def _min_numeric(rows: list[dict[str, Any]], field_name: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            raw = row.get(field_name)
            if raw is None:
                continue
            values.append(float(raw))
        except Exception:
            continue
    if not values:
        return None
    return min(values)


def _run_catalog_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    shaped = [
        {
            "run_name": row.get("run_name"),
            "status": row.get("status"),
            "trainer_name": row.get("trainer_name"),
            "surface_key": row.get("surface_key"),
            "family_ref": row.get("family_ref"),
            "training_mode": row.get("training_mode"),
            "regime_mode": row.get("regime_mode"),
            "basis_scope": row.get("basis_scope"),
            "assembler_mode": row.get("assembler_mode"),
            "piecewise_gate_status": row.get("piecewise_gate_status"),
            "orthogonality_status": row.get("orthogonality_status"),
            "orthogonality_score": row.get("orthogonality_score"),
            "pair_abs_corr_mean": row.get("pair_abs_corr_mean"),
            "residual_gain_mean": row.get("residual_gain_mean"),
            "semantic_dedup_status": row.get("semantic_dedup_status"),
            "semantic_unique_ratio": row.get("semantic_unique_ratio"),
            "gate_basis_count": row.get("gate_basis_count"),
            "selected_regime_count": row.get("selected_regime_count"),
            "basis_count": row.get("basis_count"),
            "output_expression_count": row.get("output_expression_count"),
            "artifact_id": row.get("artifact_id"),
            "fold_count": row.get("fold_count"),
            "rmse_mean": row.get("rmse_mean"),
            "rmse_std": row.get("rmse_std"),
            "coverage_error_mean": row.get("coverage_error_mean"),
            "exact_basis_hit_score": row.get("exact_basis_hit_score"),
            "exact_term_recovery_score": row.get("exact_term_recovery_score"),
            "outer_objective_score": row.get("outer_objective_score"),
            "interval_score_mean": row.get("interval_score_mean"),
            "picp_mean": row.get("picp_mean"),
            "run_id": row.get("run_id"),
        }
        for row in rows
    ]
    return pd.DataFrame(
        shaped,
        columns=[
            "run_name",
            "status",
            "trainer_name",
            "surface_key",
            "family_ref",
            "training_mode",
            "regime_mode",
            "basis_scope",
            "assembler_mode",
            "piecewise_gate_status",
            "orthogonality_status",
            "orthogonality_score",
            "pair_abs_corr_mean",
            "residual_gain_mean",
            "semantic_dedup_status",
            "semantic_unique_ratio",
            "gate_basis_count",
            "selected_regime_count",
            "basis_count",
            "output_expression_count",
            "artifact_id",
            "fold_count",
            "rmse_mean",
            "rmse_std",
            "coverage_error_mean",
            "exact_basis_hit_score",
            "exact_term_recovery_score",
            "outer_objective_score",
            "interval_score_mean",
            "picp_mean",
            "run_id",
        ],
    )


def _artifact_catalog_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    shaped = [
        {
            "trainer_name": row.get("trainer_name"),
            "head_task": row.get("head_task"),
            "artifact_kind": row.get("artifact_kind"),
            "regime_mode": row.get("regime_mode"),
            "basis_scope": row.get("basis_scope"),
            "assembler_mode": row.get("assembler_mode"),
            "piecewise_gate_status": row.get("piecewise_gate_status"),
            "orthogonality_status": row.get("orthogonality_status"),
            "orthogonality_score": row.get("orthogonality_score"),
            "pair_abs_corr_mean": row.get("pair_abs_corr_mean"),
            "residual_gain_mean": row.get("residual_gain_mean"),
            "semantic_dedup_status": row.get("semantic_dedup_status"),
            "semantic_unique_ratio": row.get("semantic_unique_ratio"),
            "gate_basis_count": row.get("gate_basis_count"),
            "selected_regime_count": row.get("selected_regime_count"),
            "basis_count": row.get("basis_count"),
            "output_expression_count": row.get("output_expression_count"),
            "artifact_id": row.get("artifact_id"),
            "fold_count": row.get("fold_count"),
            "rmse_mean": row.get("rmse_mean"),
            "rmse_std": row.get("rmse_std"),
            "coverage_error_mean": row.get("coverage_error_mean"),
            "exact_basis_hit_score": row.get("exact_basis_hit_score"),
            "exact_term_recovery_score": row.get("exact_term_recovery_score"),
            "outer_objective_score": row.get("outer_objective_score"),
            "family_concentration": row.get("family_concentration"),
            "feature_concentration": row.get("feature_concentration"),
            "run_id": row.get("run_id"),
        }
        for row in rows
    ]
    return pd.DataFrame(
        shaped,
        columns=[
            "trainer_name",
            "head_task",
            "artifact_kind",
            "regime_mode",
            "basis_scope",
            "assembler_mode",
            "piecewise_gate_status",
            "orthogonality_status",
            "orthogonality_score",
            "pair_abs_corr_mean",
            "residual_gain_mean",
            "semantic_dedup_status",
            "semantic_unique_ratio",
            "gate_basis_count",
            "selected_regime_count",
            "basis_count",
            "output_expression_count",
            "artifact_id",
            "fold_count",
            "rmse_mean",
            "rmse_std",
            "coverage_error_mean",
            "exact_basis_hit_score",
            "exact_term_recovery_score",
            "outer_objective_score",
            "family_concentration",
            "feature_concentration",
            "run_id",
        ],
    )


def _selected_table_row_indices(event: Any) -> tuple[int, ...]:
    if event is None:
        return ()
    selection = getattr(event, "selection", None)
    if selection is not None:
        rows = getattr(selection, "rows", None)
        if rows is not None:
            return tuple(int(value) for value in rows)
    if isinstance(event, Mapping):
        payload = event.get("selection")
        if isinstance(payload, Mapping):
            rows = payload.get("rows")
            if isinstance(rows, Sequence):
                return tuple(int(value) for value in rows)
    return ()


def _selection_state(selected_key: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    key = str(selected_key or "").strip()
    visible_keys = [str(row.get("selection_key", "")).strip() for row in rows]
    if not key:
        return {"selected_key": "", "visible": False, "row_index": None}
    if key in visible_keys:
        return {"selected_key": key, "visible": True, "row_index": visible_keys.index(key)}
    return {"selected_key": key, "visible": False, "row_index": None}


def _result_rows_frame(rows: Sequence[Mapping[str, Any]], *, view_mode: str) -> pd.DataFrame:
    materialized = [dict(row) for row in rows]
    return _run_catalog_frame(materialized) if view_mode == "run_catalog" else _artifact_catalog_frame(materialized)


def _render_selection_float(*, selection: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], view_mode: str) -> None:
    selected_key = str(selection.get("selected_key") or "").strip()
    if not selected_key:
        return

    row_by_key = {str(row.get("selection_key", "")).strip(): row for row in rows}
    selected_row = dict(row_by_key.get(selected_key, {}))
    title = _run_option_label(selected_row) if view_mode == "run_catalog" else _artifact_option_label(selected_row)
    row_index = selection.get("row_index")
    meta = (
        f"当前选中项位于结果表格第 {int(row_index) + 1} 行 / Current selection is on result row {int(row_index) + 1}."
        if selection.get("visible", False) and isinstance(row_index, int)
        else "当前选中项不在结果表格里 / Current selection is not visible in the result table."
    )

    info_col, prev_col, next_col = st.columns((1.8, 0.6, 0.6))
    with info_col:
        st.markdown("**当前选中 Current Selection**")
        st.caption(title or selected_key)
        st.code(selected_key, language="text")
        st.caption(meta)

    if selection.get("visible", False) and isinstance(row_index, int) and len(rows) > 1:
        prev_index = max(0, int(row_index) - 1)
        next_index = min(len(rows) - 1, int(row_index) + 1)
        prev_key = str(rows[prev_index].get("selection_key") or "").strip()
        next_key = str(rows[next_index].get("selection_key") or "").strip()
        with prev_col:
            if st.button("上一项 / Previous", use_container_width=True, disabled=int(row_index) <= 0, key=f"experiment_prev::{view_mode}"):
                st.session_state["mlblack_experiment_dashboard_selected"] = prev_key
                st.rerun()
        with next_col:
            if st.button("下一项 / Next", use_container_width=True, disabled=int(row_index) >= len(rows) - 1, key=f"experiment_next::{view_mode}"):
                st.session_state["mlblack_experiment_dashboard_selected"] = next_key
                st.rerun()


def _render_results_table(*, rows: Sequence[Mapping[str, Any]], view_mode: str) -> str:
    selected_key = str(st.session_state.get("mlblack_experiment_dashboard_selected") or "").strip()
    if not rows:
        st.info("当前筛选条件下没有结果 / No results matched the current filters.")
        return ""

    frame = _result_rows_frame(rows, view_mode=view_mode)
    try:
        table_event = st.dataframe(
            frame,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key=f"mlblack_experiment_results::{view_mode}",
        )
        selected_rows = _selected_table_row_indices(table_event)
        if selected_rows:
            index = int(selected_rows[0])
            if 0 <= index < len(rows):
                selected_key = str(rows[index].get("selection_key") or "").strip()
    except Exception:
        st.dataframe(frame, width="stretch", hide_index=True)

    if not selected_key:
        selected_key = str(rows[0].get("selection_key") or "").strip()
    return selected_key


def _run_option_label(row: dict[str, Any]) -> str:
    run_name = str(row.get("run_name") or row.get("run_id") or "").strip()
    trainer = str(row.get("trainer_name") or "N/A").strip()
    status = str(row.get("status") or "N/A").strip()
    rmse_std = _metric_text(row.get("rmse_std"))
    coverage = _metric_text(row.get("coverage_error_mean"))
    return f"{run_name} | {trainer} | {status} | rmse_std={rmse_std} | coverage_error_mean={coverage}"


def _artifact_option_key(row: dict[str, Any]) -> str:
    return f"{str(row.get('run_id') or '')}::{str(row.get('artifact_id') or '')}"


def _artifact_option_label(row: dict[str, Any]) -> str:
    artifact_id = str(row.get("artifact_id") or "artifact").strip()
    trainer = str(row.get("trainer_name") or "N/A").strip()
    head_task = str(row.get("head_task") or "N/A").strip()
    rmse_std = _metric_text(row.get("rmse_std"))
    coverage = _metric_text(row.get("coverage_error_mean"))
    return f"{artifact_id} | {trainer} | head={head_task} | rmse_std={rmse_std} | coverage_error_mean={coverage}"


def _show_json_block(payload: object, *, empty_message: str) -> None:
    if isinstance(payload, dict) and payload:
        st.json(payload, expanded=False)
        return
    if isinstance(payload, list) and payload:
        st.json(payload, expanded=False)
        return
    st.info(empty_message)


def _show_runtime_contract_layers(
    *,
    surface_record: object,
    assembly_record: object,
    run_record: object,
    artifact_record: object,
) -> None:
    st.markdown("**运行合同层 Runtime Contract Layers**")
    st.markdown("`SurfaceRecord`")
    _show_json_block(surface_record, empty_message="当前条目还没有已落库的 SurfaceRecord / No materialized SurfaceRecord yet.")
    st.markdown("`AssemblyRecord`")
    _show_json_block(assembly_record, empty_message="当前条目还没有已落库的 AssemblyRecord / No materialized AssemblyRecord yet.")
    st.markdown("`RunRecord`")
    _show_json_block(run_record, empty_message="当前条目还没有已落库的 RunRecord / No materialized RunRecord yet.")
    st.markdown("`ArtifactRecord`")
    _show_json_block(artifact_record, empty_message="当前条目还没有已落库的 ArtifactRecord / No materialized ArtifactRecord yet.")


def _set_page_config() -> None:
    st.set_page_config(page_title="mlblack Experiments", page_icon="ME", layout="wide", initial_sidebar_state="collapsed")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display: none !important;}
        .block-container {padding-top: 1.02rem; padding-bottom: 1.25rem; max-width: 1560px;}
        .experiment-hero {
            background: linear-gradient(135deg, #fff2dd 0%, #f8ddb0 46%, #e9a34f 100%);
            border: 1px solid rgba(122, 79, 30, 0.14);
            border-radius: 24px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(79, 55, 27, 0.08);
        }
        .experiment-hero-head {display: flex; align-items: center; gap: 0.88rem; margin-bottom: 0.38rem;}
        .experiment-brand {display: flex; align-items: center; gap: 0.88rem;}
        .experiment-icon {
            width: 48px;
            height: 48px;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(122, 71, 16, 0.96), rgba(177, 101, 19, 0.96));
            color: #fff3df;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            font-weight: 900;
            letter-spacing: 0.06em;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 12px 24px rgba(112, 66, 14, 0.16);
        }
        .experiment-kicker {font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #7b4f1d; font-weight: 800;}
        .experiment-title {font-size: 2.0rem; line-height: 1.05; color: #2d1e12; font-weight: 800; margin: 0.18rem 0 0.34rem 0;}
        .experiment-sub {font-size: 0.96rem; color: #59442d; max-width: 82ch;}
        .experiment-inline-filters {
            border: 1px solid rgba(90, 62, 28, 0.12);
            border-radius: 18px;
            padding: 0.2rem 0.32rem 0.32rem 0.32rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(249,245,239,0.96));
            margin-bottom: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _view_mode_label(value: str) -> str:
    return "运行目录 / Run Catalog" if str(value) == "run_catalog" else "产物目录 / Artifact Catalog"


def main() -> None:
    parser = argparse.ArgumentParser(description="mlblack experiment dashboard")
    default_info = experiment_db_config_info()
    parser.add_argument(
        "--db",
        type=str,
        default=str(default_info.get("db_target") or resolve_experiment_db_target()),
        help="Experiment DB target. Accepts a sqlite path or postgresql://... URL. Defaults to experiment/db.toml, env, catalog fallback, then local sqlite.",
    )
    parser.add_argument("--limit", type=int, default=500, help="Max runs to list")
    args, _ = parser.parse_known_args()

    _set_page_config()
    _inject_style()
    query_params, query_filters = _read_query_params(st)

    st.session_state.setdefault("mlblack_experiment_dashboard_db", str(query_params.get("db") or args.db))
    st.session_state.setdefault(
        "mlblack_experiment_dashboard_limit",
        int(query_params.get("limit", args.limit) or args.limit),
    )
    st.session_state.setdefault(
        "mlblack_experiment_dashboard_view_mode",
        str(query_params.get("view") or "run_catalog"),
    )
    st.session_state.setdefault(
        "mlblack_experiment_dashboard_selected",
        str(query_params.get("selected") or ""),
    )
    st.session_state.setdefault("run_catalog_status", _query_filter_first(query_filters, "run_status"))
    st.session_state.setdefault("run_catalog_trainer", _query_filter_first(query_filters, "run_trainer_name"))
    st.session_state.setdefault("run_catalog_fold", _fold_choice_from_query(_query_filter_first(query_filters, "run_fold_summary")))
    st.session_state.setdefault("run_catalog_surface_key", _query_filter_first(query_filters, "run_surface_key"))
    st.session_state.setdefault("run_catalog_family_ref", _query_filter_first(query_filters, "run_family_ref"))
    st.session_state.setdefault("run_catalog_assembly_signature", _query_filter_first(query_filters, "run_assembly_signature"))
    st.session_state.setdefault("run_catalog_regime_mode", _query_filter_first(query_filters, "run_regime_mode"))
    st.session_state.setdefault("run_catalog_basis_scope", _query_filter_first(query_filters, "run_basis_scope"))
    st.session_state.setdefault("run_catalog_assembler_mode", _query_filter_first(query_filters, "run_assembler_mode"))
    st.session_state.setdefault(
        "run_catalog_piecewise_gate_status",
        _query_filter_first(query_filters, "run_piecewise_gate_status"),
    )
    st.session_state.setdefault(
        "run_catalog_orthogonality_status",
        _query_filter_first(query_filters, "run_orthogonality_status"),
    )
    st.session_state.setdefault(
        "run_catalog_residual_complementarity_status",
        _query_filter_first(query_filters, "run_residual_complementarity_status"),
    )
    st.session_state.setdefault(
        "run_catalog_semantic_dedup_status",
        _query_filter_first(query_filters, "run_semantic_dedup_status"),
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="run_catalog_rmse_std",
        raw_value=_query_filter_first(query_filters, "run_rmse_std_lte"),
        default_value=1.0,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="run_catalog_coverage_error",
        raw_value=_query_filter_first(query_filters, "run_coverage_error_mean_lte"),
        default_value=0.05,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="run_catalog_exact_basis_hit_score",
        raw_value=_query_filter_first(query_filters, "run_exact_basis_hit_score_gte"),
        default_value=0.5,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="run_catalog_exact_term_recovery_score",
        raw_value=_query_filter_first(query_filters, "run_exact_term_recovery_score_gte"),
        default_value=0.5,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="run_catalog_outer_objective_score",
        raw_value=_query_filter_first(query_filters, "run_outer_objective_score_gte"),
        default_value=0.5,
    )
    st.session_state.setdefault("artifact_catalog_trainer", _query_filter_first(query_filters, "artifact_trainer_name"))
    st.session_state.setdefault("artifact_catalog_head", _query_filter_first(query_filters, "artifact_head_task"))
    st.session_state.setdefault("artifact_catalog_fold", _fold_choice_from_query(_query_filter_first(query_filters, "artifact_fold_summary")))
    st.session_state.setdefault("artifact_catalog_regime_mode", _query_filter_first(query_filters, "artifact_regime_mode"))
    st.session_state.setdefault("artifact_catalog_basis_scope", _query_filter_first(query_filters, "artifact_basis_scope"))
    st.session_state.setdefault(
        "artifact_catalog_assembler_mode",
        _query_filter_first(query_filters, "artifact_assembler_mode"),
    )
    st.session_state.setdefault(
        "artifact_catalog_piecewise_gate_status",
        _query_filter_first(query_filters, "artifact_piecewise_gate_status"),
    )
    st.session_state.setdefault(
        "artifact_catalog_orthogonality_status",
        _query_filter_first(query_filters, "artifact_orthogonality_status"),
    )
    st.session_state.setdefault(
        "artifact_catalog_residual_complementarity_status",
        _query_filter_first(query_filters, "artifact_residual_complementarity_status"),
    )
    st.session_state.setdefault(
        "artifact_catalog_semantic_dedup_status",
        _query_filter_first(query_filters, "artifact_semantic_dedup_status"),
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="artifact_catalog_rmse_std",
        raw_value=_query_filter_first(query_filters, "artifact_rmse_std_lte"),
        default_value=1.0,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="artifact_catalog_coverage_error",
        raw_value=_query_filter_first(query_filters, "artifact_coverage_error_mean_lte"),
        default_value=0.05,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="artifact_catalog_exact_basis_hit_score",
        raw_value=_query_filter_first(query_filters, "artifact_exact_basis_hit_score_gte"),
        default_value=0.5,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="artifact_catalog_exact_term_recovery_score",
        raw_value=_query_filter_first(query_filters, "artifact_exact_term_recovery_score_gte"),
        default_value=0.5,
    )
    _apply_threshold_query_state(
        st.session_state,
        key_prefix="artifact_catalog_outer_objective_score",
        raw_value=_query_filter_first(query_filters, "artifact_outer_objective_score_gte"),
        default_value=0.5,
    )

    st.markdown(
        (
            "<div class='experiment-hero'>"
            "<div class='experiment-hero-head'>"
            "<div class='experiment-brand'>"
            "<div class='experiment-icon'>ME</div>"
            "<div>"
            "<div class='experiment-kicker'>mlblack experiments</div>"
            "<div class='experiment-title'>实验结果 / 产物双视图工作台 Experiment Runs / Artifacts</div>"
            "</div></div></div>"
            "<div class='experiment-sub'>"
            "顶部统一控制实验库目标、结果上限与视图模式；优先读取 materialized run / artifact catalog，"
            "缺失时自动回退到 legacy run / event / metric / trace 浏览。"
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )

    top_cols = st.columns((1.45, 0.65, 0.8, 0.95))
    raw_db_target = top_cols[0].text_input("实验库 / Experiment DB", key="mlblack_experiment_dashboard_db")
    run_limit = int(
        top_cols[1].number_input(
            "结果上限 / Result Limit",
            min_value=10,
            max_value=5000,
            value=int(st.session_state.get("mlblack_experiment_dashboard_limit", args.limit)),
            key="mlblack_experiment_dashboard_limit",
        )
    )
    refresh_key = "mlblack_experiment_dashboard_refresh_nonce"
    if refresh_key not in st.session_state:
        st.session_state[refresh_key] = 0
    if top_cols[2].button("刷新 DB 缓存 / Refresh DB Cache", use_container_width=True):
        st.session_state[refresh_key] = int(st.session_state.get(refresh_key, 0)) + 1
    refresh_nonce = int(st.session_state.get(refresh_key, 0))
    view_mode = top_cols[3].selectbox(
        "视图 / View",
        options=("run_catalog", "artifact_catalog"),
        index=("run_catalog", "artifact_catalog").index(
            str(st.session_state.get("mlblack_experiment_dashboard_view_mode", "run_catalog"))
            if str(st.session_state.get("mlblack_experiment_dashboard_view_mode", "run_catalog")) in {"run_catalog", "artifact_catalog"}
            else "run_catalog"
        ),
        format_func=_view_mode_label,
        key="mlblack_experiment_dashboard_view_mode",
    )

    st.caption("如果 flow 已启用 experiment_tracker，建议把 run / artifact catalog materialize 到同一个实验库。")

    db_path = resolve_experiment_db_target(raw_db_target)
    db_target_info = normalize_experiment_db_target(db_path)
    try:
        summary = experiment_tracker_summary(str(db_path))
        table_state = _catalog_table_state(str(db_path), refresh_nonce)
        filter_values = _load_catalog_filter_values(str(db_path), refresh_nonce)
    except Exception as exc:
        st.warning(f"实验库不可连接 / Experiment DB is unavailable: {db_target_info.safe_label}")
        st.caption(str(exc))
        st.code(
            "在 flow 配置里启用 experiment_tracker:\n"
            '"capabilities": [{"key":"experiment_tracker","params":{"db_path":"runs/experiments.sqlite3"}}]'
        )
        return

    run_catalog_ready = bool(table_state.get("run_catalog", False))
    artifact_catalog_ready = bool(table_state.get("artifact_catalog", False))

    stat_cols = st.columns(4)
    stat_cols[0].metric("当前视图 / View", _view_mode_label(view_mode))
    stat_cols[1].metric("目录就绪 / Catalog Ready", f"run={int(run_catalog_ready)} / artifact={int(artifact_catalog_ready)}")
    stat_cols[2].metric("数据库状态 / DB Status", f"已连接 / Connected · {summary.get('backend', db_target_info.backend)}")
    stat_cols[3].metric("结果上限 / Result Limit", str(run_limit))

    selected_run_id: str | None = None
    selected_artifact_id: str | None = None
    status_filter: str | None = None
    trainer_filter: str | None = None
    surface_key_filter: str | None = None
    family_ref_filter: str | None = None
    assembly_signature_filter: str | None = None
    regime_mode_filter: str | None = None
    basis_scope_filter: str | None = None
    assembler_mode_filter: str | None = None
    piecewise_gate_status_filter: str | None = None
    orthogonality_status_filter: str | None = None
    residual_complementarity_status_filter: str | None = None
    semantic_dedup_status_filter: str | None = None
    artifact_trainer_filter: str | None = None
    head_task_filter: str | None = None
    artifact_regime_mode_filter: str | None = None
    artifact_basis_scope_filter: str | None = None
    artifact_assembler_mode_filter: str | None = None
    artifact_piecewise_gate_status_filter: str | None = None
    artifact_orthogonality_status_filter: str | None = None
    artifact_residual_complementarity_status_filter: str | None = None
    artifact_semantic_dedup_status_filter: str | None = None
    max_rmse_std: float | None = None
    max_coverage_error: float | None = None
    min_exact_basis_hit_score: float | None = None
    min_exact_term_recovery_score: float | None = None
    min_outer_objective_score: float | None = None
    fold_choice: str = "不限 / Any"

    if view_mode == "run_catalog" and run_catalog_ready:
        with st.container():
            st.markdown("<div class='experiment-inline-filters'>", unsafe_allow_html=True)
            with st.expander("运行结果筛选 / Run Filters", expanded=False):
                filter_cols = st.columns(5)
                with filter_cols[0]:
                    status_filter = _optional_exact_filter("状态 / Status", list(filter_values.get("run_status", [])), key="run_catalog_status")
                with filter_cols[1]:
                    trainer_filter = _optional_exact_filter("训练器 / Trainer", list(filter_values.get("run_trainer_name", [])), key="run_catalog_trainer")
                with filter_cols[2]:
                    fold_choice = st.selectbox("fold_summary", ["不限 / Any", "有 / Present", "无 / Absent"], index=0, key="run_catalog_fold")
                with filter_cols[3]:
                    max_rmse_std = _metric_threshold_input("rmse_std", key_prefix="run_catalog_rmse_std", default_value=1.0)
                with filter_cols[4]:
                    max_coverage_error = _metric_threshold_input("coverage_error_mean", key_prefix="run_catalog_coverage_error", default_value=0.05)
                contract_filter_cols = st.columns(3)
                with contract_filter_cols[0]:
                    surface_key_filter = _optional_exact_filter(
                        "surface_key",
                        list(filter_values.get("run_surface_key", [])),
                        key="run_catalog_surface_key",
                    )
                with contract_filter_cols[1]:
                    family_ref_filter = _optional_exact_filter(
                        "family_ref",
                        list(filter_values.get("run_family_ref", [])),
                        key="run_catalog_family_ref",
                    )
                with contract_filter_cols[2]:
                    assembly_signature_filter = st.text_input(
                        "assembly_signature",
                        value="",
                        key="run_catalog_assembly_signature",
                        help="精确匹配已落库的 AssemblyRecord 签名 / Exact match against the materialized AssemblyRecord signature.",
                    ).strip() or None
                structure_filter_cols = st.columns(4)
                with structure_filter_cols[0]:
                    regime_mode_filter = _optional_exact_filter(
                        "regime_mode",
                        list(filter_values.get("run_regime_mode", [])),
                        key="run_catalog_regime_mode",
                    )
                with structure_filter_cols[1]:
                    basis_scope_filter = _optional_exact_filter(
                        "basis_scope",
                        list(filter_values.get("run_basis_scope", [])),
                        key="run_catalog_basis_scope",
                    )
                with structure_filter_cols[2]:
                    assembler_mode_filter = _optional_exact_filter(
                        "assembler_mode",
                        list(filter_values.get("run_assembler_mode", [])),
                        key="run_catalog_assembler_mode",
                    )
                with structure_filter_cols[3]:
                    piecewise_gate_status_filter = _optional_exact_filter(
                        "piecewise_gate_status",
                        list(filter_values.get("run_piecewise_gate_status", [])),
                        key="run_catalog_piecewise_gate_status",
                    )
                orthogonal_filter_cols = st.columns(3)
                with orthogonal_filter_cols[0]:
                    orthogonality_status_filter = _optional_exact_filter(
                        "orthogonality_status",
                        list(filter_values.get("run_orthogonality_status", [])),
                        key="run_catalog_orthogonality_status",
                    )
                with orthogonal_filter_cols[1]:
                    residual_complementarity_status_filter = _optional_exact_filter(
                        "residual_complementarity_status",
                        list(filter_values.get("run_residual_complementarity_status", [])),
                        key="run_catalog_residual_complementarity_status",
                    )
                with orthogonal_filter_cols[2]:
                    semantic_dedup_status_filter = _optional_exact_filter(
                        "semantic_dedup_status",
                        list(filter_values.get("run_semantic_dedup_status", [])),
                        key="run_catalog_semantic_dedup_status",
                    )
                benchmark_filter_cols = st.columns(3)
                with benchmark_filter_cols[0]:
                    min_exact_basis_hit_score = _metric_floor_input(
                        "exact_basis_hit_score",
                        key_prefix="run_catalog_exact_basis_hit_score",
                        default_value=0.5,
                    )
                with benchmark_filter_cols[1]:
                    min_exact_term_recovery_score = _metric_floor_input(
                        "exact_term_recovery_score",
                        key_prefix="run_catalog_exact_term_recovery_score",
                        default_value=0.5,
                    )
                with benchmark_filter_cols[2]:
                    min_outer_objective_score = _metric_floor_input(
                        "outer_objective_score",
                        key_prefix="run_catalog_outer_objective_score",
                        default_value=0.5,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        run_catalog_rows = [
            {
                **dict(row),
                "selection_key": _selection_run_key(str(row.get("run_id") or "").strip()),
            }
            for row in _load_run_catalog_cached(
                str(db_path),
                status_filter,
                trainer_filter,
                surface_key_filter,
                family_ref_filter,
                assembly_signature_filter,
                regime_mode_filter,
                basis_scope_filter,
                assembler_mode_filter,
                piecewise_gate_status_filter,
                orthogonality_status_filter,
                residual_complementarity_status_filter,
                semantic_dedup_status_filter,
                _fold_filter_value(str(fold_choice)),
                max_rmse_std,
                max_coverage_error,
                min_exact_basis_hit_score,
                min_exact_term_recovery_score,
                min_outer_objective_score,
                run_limit,
                refresh_nonce,
            )
            if str(row.get("run_id") or "").strip()
        ]
        st.subheader("运行目录 / Run Catalog")
        if not run_catalog_rows:
            st.info("当前筛选条件下没有命中的运行目录条目 / No run catalog rows matched the current filters.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("命中运行数 / Runs", str(len(run_catalog_rows)))
            c2.metric("含 fold_summary / With Fold Summary", str(_count_fold_summary_rows(run_catalog_rows)))
            c3.metric("最优 rmse_std / Best rmse_std", _metric_text(_min_numeric(run_catalog_rows, "rmse_std")))

            row_by_key = {str(row.get("selection_key") or "").strip(): row for row in run_catalog_rows}
            selected_payload = _decode_selection_key(str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""))
            default_run_key = (
                _selection_run_key(str(selected_payload.get("run_id")))
                if selected_payload and str(selected_payload.get("kind")) == "run"
                else str(st.session_state.get("mlblack_experiment_dashboard_selected") or "")
            )
            if default_run_key not in row_by_key:
                default_run_key = next(iter(row_by_key.keys()), "")
            if default_run_key:
                st.session_state["mlblack_experiment_dashboard_selected"] = default_run_key

            selection = _selection_state(str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""), run_catalog_rows)
            _render_selection_float(selection=selection, rows=run_catalog_rows, view_mode="run_catalog")
            selected_run_key = _render_results_table(rows=run_catalog_rows, view_mode="run_catalog")
            if selected_run_key and selected_run_key != str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""):
                st.session_state["mlblack_experiment_dashboard_selected"] = selected_run_key

            selected_run_key = str(st.session_state.get("mlblack_experiment_dashboard_selected") or "").strip()
            selected_run_row = dict(row_by_key.get(selected_run_key, {}))
            selected_run_id = str(selected_run_row.get("run_id") or "").strip()
            st.session_state["mlblack_experiment_dashboard_run_id"] = selected_run_id

            selected_run_catalog = _show_run_catalog_cached(str(db_path), selected_run_id, refresh_nonce) or selected_run_row
            linked_artifact_catalog = None
            linked_artifact_id = str(selected_run_catalog.get("artifact_id") or "").strip()
            if linked_artifact_id:
                linked_artifact_catalog = _show_artifact_catalog_cached(
                    str(db_path),
                    selected_run_id,
                    linked_artifact_id,
                    refresh_nonce,
                )
            st.subheader("运行详情 / Run Detail")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("状态 / Status", str(selected_run_catalog.get("status") or "-"))
            m2.metric("训练器 / Trainer", str(selected_run_catalog.get("trainer_name") or "-"))
            m3.metric("fold_count", str(selected_run_catalog.get("fold_count") or "-"))
            m4.metric("rmse_std", _metric_text(selected_run_catalog.get("rmse_std")))
            m5.metric("coverage_error_mean", _metric_text(selected_run_catalog.get("coverage_error_mean")))
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("orthogonality_score", _metric_text(selected_run_catalog.get("orthogonality_score")))
            o2.metric("pair_abs_corr_mean", _metric_text(selected_run_catalog.get("pair_abs_corr_mean")))
            o3.metric("residual_gain_mean", _metric_text(selected_run_catalog.get("residual_gain_mean")))
            o4.metric("semantic_unique_ratio", _metric_text(selected_run_catalog.get("semantic_unique_ratio")))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("exact_basis_hit_score", _metric_text(selected_run_catalog.get("exact_basis_hit_score")))
            b2.metric("exact_term_recovery_score", _metric_text(selected_run_catalog.get("exact_term_recovery_score")))
            b3.metric("outer_objective_score", _metric_text(selected_run_catalog.get("outer_objective_score")))
            b4.metric("inner_fit_score", _metric_text(selected_run_catalog.get("inner_fit_score")))
            st.caption(
                f"run_id={selected_run_id} | artifact_id={selected_run_catalog.get('artifact_id') or '-'} | "
                f"training_mode={selected_run_catalog.get('training_mode') or '-'}"
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**fold_summary**")
                _show_json_block(
                    selected_run_catalog.get("fold_summary_json"),
                    empty_message="当前 run 还没有已落库的 fold_summary / No materialized fold_summary yet.",
                )
                st.markdown("**产物目录投影 / Artifact Catalog Projection**")
                _show_json_block(
                    selected_run_catalog.get("artifact_catalog_json"),
                    empty_message="当前 run 还没有产物目录投影 / No artifact catalog projection yet.",
                )
            with right:
                _show_runtime_contract_layers(
                    surface_record=selected_run_catalog.get("surface_record_json"),
                    assembly_record=selected_run_catalog.get("assembly_record_json"),
                    run_record=selected_run_catalog.get("run_record_json"),
                    artifact_record=None if linked_artifact_catalog is None else linked_artifact_catalog.get("artifact_record_json"),
                )
                st.markdown("**Warm Start / 兼容性漂移 Compatibility Drift**")
                _show_json_block(
                    selected_run_catalog.get("compatibility_drift_json"),
                    empty_message="当前 run 还没有兼容性漂移信息 / No compatibility drift recorded yet.",
                )
                st.markdown("**搜索族签名合同 / Search Family Signature Contracts**")
                _show_json_block(
                    selected_run_catalog.get("search_family_signature_contracts_json"),
                    empty_message="当前 run 还没有搜索族签名合同 / No search family signature contracts recorded yet.",
                )
                st.markdown("**真值合同恢复 / Truth Contract Recovery**")
                _show_json_block(
                    selected_run_catalog.get("truth_contract_recovery_json"),
                    empty_message="当前 run 还没有 truth contract recovery / No truth contract recovery recorded yet.",
                )
                st.markdown("**外层目标 / Orthogonal Search Objective**")
                _show_json_block(
                    selected_run_catalog.get("orthogonal_search_objective_json"),
                    empty_message="当前 run 还没有 orthogonal search objective / No orthogonal search objective recorded yet.",
                )

            with st.expander("展开 task_signature / symbolic_family / report", expanded=False):
                st.markdown("**task_signature**")
                _show_json_block(selected_run_catalog.get("task_signature_json"), empty_message="当前 run 没有 task_signature / No task_signature yet.")
                st.markdown("**symbolic_family**")
                _show_json_block(selected_run_catalog.get("symbolic_family_json"), empty_message="当前 run 没有 symbolic_family 投影 / No symbolic_family projection yet.")
                st.markdown("**report 投影 / Report Projection**")
                _show_json_block(selected_run_catalog.get("report_json"), empty_message="当前 run 没有 report 投影 / No report projection yet.")
            with st.expander("结构块 / Structure Blocks", expanded=False):
                st.markdown("**regime_structure**")
                _show_json_block(selected_run_catalog.get("regime_structure_json"), empty_message="当前 run 还没有 regime_structure / No regime_structure recorded yet.")
                st.markdown("**basis_structure**")
                _show_json_block(selected_run_catalog.get("basis_structure_json"), empty_message="当前 run 还没有 basis_structure / No basis_structure recorded yet.")
                st.markdown("**assembler_structure**")
                _show_json_block(selected_run_catalog.get("assembler_structure_json"), empty_message="当前 run 还没有 assembler_structure / No assembler_structure recorded yet.")
                st.markdown("**piecewise_gate_basis**")
                _show_json_block(selected_run_catalog.get("piecewise_gate_basis_json"), empty_message="当前 run 还没有 piecewise_gate_basis / No piecewise_gate_basis recorded yet.")
    elif view_mode == "artifact_catalog" and artifact_catalog_ready:
        with st.container():
            st.markdown("<div class='experiment-inline-filters'>", unsafe_allow_html=True)
            with st.expander("产物结果筛选 / Artifact Filters", expanded=False):
                filter_cols = st.columns(5)
                with filter_cols[0]:
                    artifact_trainer_filter = _optional_exact_filter("训练器 / Trainer", list(filter_values.get("artifact_trainer_name", [])), key="artifact_catalog_trainer")
                with filter_cols[1]:
                    head_task_filter = _optional_exact_filter("head_task", list(filter_values.get("artifact_head_task", [])), key="artifact_catalog_head")
                with filter_cols[2]:
                    fold_choice = st.selectbox("fold_summary", ["不限 / Any", "有 / Present", "无 / Absent"], index=0, key="artifact_catalog_fold")
                with filter_cols[3]:
                    max_rmse_std = _metric_threshold_input("rmse_std", key_prefix="artifact_catalog_rmse_std", default_value=1.0)
                with filter_cols[4]:
                    max_coverage_error = _metric_threshold_input("coverage_error_mean", key_prefix="artifact_catalog_coverage_error", default_value=0.05)
                structure_filter_cols = st.columns(4)
                with structure_filter_cols[0]:
                    artifact_regime_mode_filter = _optional_exact_filter(
                        "regime_mode",
                        list(filter_values.get("artifact_regime_mode", [])),
                        key="artifact_catalog_regime_mode",
                    )
                with structure_filter_cols[1]:
                    artifact_basis_scope_filter = _optional_exact_filter(
                        "basis_scope",
                        list(filter_values.get("artifact_basis_scope", [])),
                        key="artifact_catalog_basis_scope",
                    )
                with structure_filter_cols[2]:
                    artifact_assembler_mode_filter = _optional_exact_filter(
                        "assembler_mode",
                        list(filter_values.get("artifact_assembler_mode", [])),
                        key="artifact_catalog_assembler_mode",
                    )
                with structure_filter_cols[3]:
                    artifact_piecewise_gate_status_filter = _optional_exact_filter(
                        "piecewise_gate_status",
                        list(filter_values.get("artifact_piecewise_gate_status", [])),
                        key="artifact_catalog_piecewise_gate_status",
                    )
                orthogonal_filter_cols = st.columns(3)
                with orthogonal_filter_cols[0]:
                    artifact_orthogonality_status_filter = _optional_exact_filter(
                        "orthogonality_status",
                        list(filter_values.get("artifact_orthogonality_status", [])),
                        key="artifact_catalog_orthogonality_status",
                    )
                with orthogonal_filter_cols[1]:
                    artifact_residual_complementarity_status_filter = _optional_exact_filter(
                        "residual_complementarity_status",
                        list(filter_values.get("artifact_residual_complementarity_status", [])),
                        key="artifact_catalog_residual_complementarity_status",
                    )
                with orthogonal_filter_cols[2]:
                    artifact_semantic_dedup_status_filter = _optional_exact_filter(
                        "semantic_dedup_status",
                        list(filter_values.get("artifact_semantic_dedup_status", [])),
                        key="artifact_catalog_semantic_dedup_status",
                    )
                benchmark_filter_cols = st.columns(3)
                with benchmark_filter_cols[0]:
                    min_exact_basis_hit_score = _metric_floor_input(
                        "exact_basis_hit_score",
                        key_prefix="artifact_catalog_exact_basis_hit_score",
                        default_value=0.5,
                    )
                with benchmark_filter_cols[1]:
                    min_exact_term_recovery_score = _metric_floor_input(
                        "exact_term_recovery_score",
                        key_prefix="artifact_catalog_exact_term_recovery_score",
                        default_value=0.5,
                    )
                with benchmark_filter_cols[2]:
                    min_outer_objective_score = _metric_floor_input(
                        "outer_objective_score",
                        key_prefix="artifact_catalog_outer_objective_score",
                        default_value=0.5,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        artifact_catalog_rows = [
            {
                **dict(row),
                "selection_key": _selection_artifact_key(
                    str(row.get("run_id") or "").strip(),
                    str(row.get("artifact_id") or "").strip(),
                ),
            }
            for row in _load_artifact_catalog_cached(
                str(db_path),
                artifact_trainer_filter,
                head_task_filter,
                artifact_regime_mode_filter,
                artifact_basis_scope_filter,
                artifact_assembler_mode_filter,
                artifact_piecewise_gate_status_filter,
                artifact_orthogonality_status_filter,
                artifact_residual_complementarity_status_filter,
                artifact_semantic_dedup_status_filter,
                _fold_filter_value(str(fold_choice)),
                max_rmse_std,
                max_coverage_error,
                min_exact_basis_hit_score,
                min_exact_term_recovery_score,
                min_outer_objective_score,
                run_limit,
                refresh_nonce,
            )
            if str(row.get("run_id") or "").strip() and str(row.get("artifact_id") or "").strip()
        ]
        st.subheader("产物目录 / Artifact Catalog")
        if not artifact_catalog_rows:
            st.info("当前筛选条件下没有命中的产物目录条目 / No artifact catalog rows matched the current filters.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("命中产物数 / Artifacts", str(len(artifact_catalog_rows)))
            c2.metric("含 fold_summary / With Fold Summary", str(_count_fold_summary_rows(artifact_catalog_rows)))
            c3.metric("最优 coverage_error_mean / Best coverage_error_mean", _metric_text(_min_numeric(artifact_catalog_rows, "coverage_error_mean")))

            row_by_key = {str(row.get("selection_key") or "").strip(): row for row in artifact_catalog_rows}
            selected_payload = _decode_selection_key(str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""))
            default_artifact_key = (
                _selection_artifact_key(
                    str(selected_payload.get("run_id")),
                    str(selected_payload.get("artifact_id")),
                )
                if selected_payload and str(selected_payload.get("kind")) == "artifact"
                else str(st.session_state.get("mlblack_experiment_dashboard_selected") or "")
            )
            if default_artifact_key not in row_by_key:
                default_artifact_key = next(iter(row_by_key.keys()), "")
            if default_artifact_key:
                st.session_state["mlblack_experiment_dashboard_selected"] = default_artifact_key

            selection = _selection_state(str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""), artifact_catalog_rows)
            _render_selection_float(selection=selection, rows=artifact_catalog_rows, view_mode="artifact_catalog")
            selected_artifact_key = _render_results_table(rows=artifact_catalog_rows, view_mode="artifact_catalog")
            if selected_artifact_key and selected_artifact_key != str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""):
                st.session_state["mlblack_experiment_dashboard_selected"] = selected_artifact_key

            selected_artifact_key = str(st.session_state.get("mlblack_experiment_dashboard_selected") or "").strip()
            selected_artifact_row = dict(row_by_key.get(selected_artifact_key, {}))
            selected_run_id = str(selected_artifact_row.get("run_id") or "").strip()
            selected_artifact_id = str(selected_artifact_row.get("artifact_id") or "").strip()
            st.session_state["mlblack_experiment_dashboard_artifact_key"] = selected_artifact_key
            selected_artifact_catalog = _show_artifact_catalog_cached(
                str(db_path),
                selected_run_id,
                selected_artifact_id,
                refresh_nonce,
            ) or selected_artifact_row
            linked_run_catalog = _show_run_catalog_cached(str(db_path), selected_run_id, refresh_nonce) or {}

            st.subheader("产物详情 / Artifact Detail")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("训练器 / Trainer", str(selected_artifact_catalog.get("trainer_name") or "-"))
            m2.metric("head_task", str(selected_artifact_catalog.get("head_task") or "-"))
            m3.metric("fold_count", str(selected_artifact_catalog.get("fold_count") or "-"))
            m4.metric("rmse_std", _metric_text(selected_artifact_catalog.get("rmse_std")))
            m5.metric("coverage_error_mean", _metric_text(selected_artifact_catalog.get("coverage_error_mean")))
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("orthogonality_score", _metric_text(selected_artifact_catalog.get("orthogonality_score")))
            o2.metric("pair_abs_corr_mean", _metric_text(selected_artifact_catalog.get("pair_abs_corr_mean")))
            o3.metric("residual_gain_mean", _metric_text(selected_artifact_catalog.get("residual_gain_mean")))
            o4.metric("semantic_unique_ratio", _metric_text(selected_artifact_catalog.get("semantic_unique_ratio")))
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("exact_basis_hit_score", _metric_text(selected_artifact_catalog.get("exact_basis_hit_score")))
            b2.metric("exact_term_recovery_score", _metric_text(selected_artifact_catalog.get("exact_term_recovery_score")))
            b3.metric("outer_objective_score", _metric_text(selected_artifact_catalog.get("outer_objective_score")))
            b4.metric("inner_fit_score", _metric_text(selected_artifact_catalog.get("inner_fit_score")))
            st.caption(
                f"artifact_id={selected_artifact_id} | run_id={selected_run_id} | "
                f"artifact_kind={selected_artifact_catalog.get('artifact_kind') or '-'}"
            )

            left, right = st.columns(2)
            with left:
                st.markdown("**输出头语义 / Head Semantics**")
                _show_json_block(
                    selected_artifact_catalog.get("head_semantics_json"),
                    empty_message="当前 artifact 还没有输出头语义投影 / No head semantics projection yet.",
                )
                st.markdown("**复杂度指标 / Complexity Metrics**")
                _show_json_block(
                    selected_artifact_catalog.get("complexity_metrics_json"),
                    empty_message="当前 artifact 还没有复杂度指标 / No complexity metrics yet.",
                )
            with right:
                _show_runtime_contract_layers(
                    surface_record=linked_run_catalog.get("surface_record_json"),
                    assembly_record=linked_run_catalog.get("assembly_record_json"),
                    run_record=linked_run_catalog.get("run_record_json"),
                    artifact_record=selected_artifact_catalog.get("artifact_record_json"),
                )
                st.markdown("**稳定性 / fold_summary Stability**")
                _show_json_block(
                    selected_artifact_catalog.get("fold_summary_json"),
                    empty_message="当前 artifact 还没有 fold_summary / No fold_summary yet.",
                )
                st.markdown("**搜索族签名合同 / Search Family Signature Contracts**")
                _show_json_block(
                    selected_artifact_catalog.get("search_family_signature_contracts_json"),
                    empty_message="当前 artifact 还没有搜索族签名合同 / No search family signature contracts yet.",
                )
                st.markdown("**真值合同恢复 / Truth Contract Recovery**")
                _show_json_block(
                    selected_artifact_catalog.get("truth_contract_recovery_json"),
                    empty_message="当前 artifact 还没有 truth contract recovery / No truth contract recovery recorded yet.",
                )
                st.markdown("**外层目标 / Orthogonal Search Objective**")
                _show_json_block(
                    selected_artifact_catalog.get("orthogonal_search_objective_json"),
                    empty_message="当前 artifact 还没有 orthogonal search objective / No orthogonal search objective recorded yet.",
                )
            with st.expander("展开 stability_metrics / source_report", expanded=False):
                st.markdown("**stability_metrics**")
                _show_json_block(
                    selected_artifact_catalog.get("stability_metrics_json"),
                    empty_message="当前 artifact 还没有 stability_metrics / No stability_metrics yet.",
                )
                st.markdown("**source_report 投影 / Source Report Projection**")
                _show_json_block(selected_artifact_catalog.get("source_report_json"), empty_message="当前 artifact 还没有 source_report 投影 / No source_report projection yet.")
            with st.expander("结构块 / Structure Blocks", expanded=False):
                st.markdown("**regime_structure**")
                _show_json_block(selected_artifact_catalog.get("regime_structure_json"), empty_message="当前 artifact 还没有 regime_structure / No regime_structure recorded yet.")
                st.markdown("**basis_structure**")
                _show_json_block(selected_artifact_catalog.get("basis_structure_json"), empty_message="当前 artifact 还没有 basis_structure / No basis_structure recorded yet.")
                st.markdown("**assembler_structure**")
                _show_json_block(selected_artifact_catalog.get("assembler_structure_json"), empty_message="当前 artifact 还没有 assembler_structure / No assembler_structure recorded yet.")
                st.markdown("**piecewise_gate_basis**")
                _show_json_block(selected_artifact_catalog.get("piecewise_gate_basis_json"), empty_message="当前 artifact 还没有 piecewise_gate_basis / No piecewise_gate_basis recorded yet.")
    else:
        st.warning(
            "当前数据库里还没有 experiment_run_catalog / experiment_artifact_catalog 视图。"
            "页面会回退到 legacy run 浏览；如果希望使用正式实验面，请先启用 experiment_tracker 并完成 materialize。"
        )
        runs = _load_runs_cached(str(db_path), run_limit, refresh_nonce)
        if runs.empty:
            st.info("当前数据库里还没有实验 run / No experiment runs recorded yet.")
            return
        run_ids = [str(x) for x in runs["run_id"].tolist()]
        selected_run_id = st.selectbox("运行 ID / Run ID", options=run_ids, index=0, key="legacy_run_id")
        st.session_state["mlblack_experiment_dashboard_selected"] = _selection_run_key(selected_run_id)
        st.dataframe(runs, width="stretch", hide_index=True)

    field_filters: dict[str, object] = {}
    if view_mode == "run_catalog":
        field_filters = {
            "run_status": status_filter or "",
            "run_trainer_name": trainer_filter or "",
            "run_fold_summary": _fold_query_token(str(fold_choice)),
            "run_surface_key": surface_key_filter or "",
            "run_family_ref": family_ref_filter or "",
            "run_assembly_signature": assembly_signature_filter or "",
            "run_regime_mode": regime_mode_filter or "",
            "run_basis_scope": basis_scope_filter or "",
            "run_assembler_mode": assembler_mode_filter or "",
            "run_piecewise_gate_status": piecewise_gate_status_filter or "",
            "run_orthogonality_status": orthogonality_status_filter or "",
            "run_residual_complementarity_status": residual_complementarity_status_filter or "",
            "run_semantic_dedup_status": semantic_dedup_status_filter or "",
            "run_rmse_std_lte": "" if max_rmse_std is None else str(max_rmse_std),
            "run_coverage_error_mean_lte": "" if max_coverage_error is None else str(max_coverage_error),
            "run_exact_basis_hit_score_gte": "" if min_exact_basis_hit_score is None else str(min_exact_basis_hit_score),
            "run_exact_term_recovery_score_gte": "" if min_exact_term_recovery_score is None else str(min_exact_term_recovery_score),
            "run_outer_objective_score_gte": "" if min_outer_objective_score is None else str(min_outer_objective_score),
        }
    elif view_mode == "artifact_catalog":
        field_filters = {
            "artifact_trainer_name": artifact_trainer_filter or "",
            "artifact_head_task": head_task_filter or "",
            "artifact_regime_mode": artifact_regime_mode_filter or "",
            "artifact_basis_scope": artifact_basis_scope_filter or "",
            "artifact_assembler_mode": artifact_assembler_mode_filter or "",
            "artifact_piecewise_gate_status": artifact_piecewise_gate_status_filter or "",
            "artifact_orthogonality_status": artifact_orthogonality_status_filter or "",
            "artifact_residual_complementarity_status": artifact_residual_complementarity_status_filter or "",
            "artifact_semantic_dedup_status": artifact_semantic_dedup_status_filter or "",
            "artifact_fold_summary": _fold_query_token(str(fold_choice)),
            "artifact_rmse_std_lte": "" if max_rmse_std is None else str(max_rmse_std),
            "artifact_coverage_error_mean_lte": "" if max_coverage_error is None else str(max_coverage_error),
            "artifact_exact_basis_hit_score_gte": "" if min_exact_basis_hit_score is None else str(min_exact_basis_hit_score),
            "artifact_exact_term_recovery_score_gte": "" if min_exact_term_recovery_score is None else str(min_exact_term_recovery_score),
            "artifact_outer_objective_score_gte": "" if min_outer_objective_score is None else str(min_outer_objective_score),
        }
    field_filters = {str(k): v for k, v in field_filters.items() if str(v or "").strip() and str(v) != "any"}
    base_params = {
        "db": str(db_path),
        "limit": str(int(run_limit)),
        "view": str(view_mode),
        "selected": str(st.session_state.get("mlblack_experiment_dashboard_selected") or ""),
    }
    _write_query_params(st, base_params=base_params, field_filters=field_filters)
    st.text_input(
        "鐩磋揪閾炬帴",
        value=_build_deep_link_query(base_params=base_params, field_filters=field_filters),
        key="mlblack_experiment_dashboard_deeplink",
    )

    if not selected_run_id:
        return

    run_overview = _load_run_overview_cached(str(db_path), selected_run_id, refresh_nonce)
    events = _load_events_cached(str(db_path), selected_run_id, refresh_nonce)
    metrics = _load_metrics_cached(str(db_path), selected_run_id, refresh_nonce)
    trace = _load_training_trace_cached(str(db_path), selected_run_id, refresh_nonce)
    derived = _to_trace_derived_frame(trace) if not trace.empty else pd.DataFrame()
    if not derived.empty:
        active_terms, cluster_path, readout_top_terms = _replay_function_cluster(derived)
    else:
        active_terms = pd.DataFrame()
        cluster_path = pd.DataFrame()
        readout_top_terms = pd.DataFrame()

    st.markdown("---")
    st.subheader("当前运行总览 / Current Run Overview")
    if run_overview.empty:
        st.info("当前 run 没有总览记录 / No run overview recorded yet.")
    else:
        st.dataframe(run_overview, width="stretch", hide_index=True)

    st.markdown("**显示公式（y = f(x)）/ Displayed Formula**")
    lines = _formula_lines_from_readout_top_terms(readout_top_terms)
    if lines:
        st.code("\n".join(lines), language="text")
        st.caption("根据最后一次 readout top terms 反推的近似显示公式；`b` 表示偏置项 / Reconstructed from the latest readout top terms.")
    else:
        st.info("当前 run 还没有可重建的显示公式 / No reconstructed display formula yet.")

    st.markdown("**函数簇快照 / Function Cluster Snapshot**")
    c7, c8 = st.columns(2)
    with c7:
        st.caption("根据 add / replace / prune trace 回放出的当前活跃函数簇 / Active terms replayed from the trace.")
        if active_terms.empty:
            st.info("当前没有可展示的活跃项 / No active terms to display.")
        else:
            st.dataframe(active_terms, width="stretch", hide_index=True)
    with c8:
        st.caption("函数簇规模和操作路径的演化快照 / Evolution snapshot of term counts and operation path.")
        if cluster_path.empty:
            st.info("当前没有函数簇演化轨迹 / No cluster evolution trace yet.")
        else:
            st.dataframe(cluster_path, width="stretch", hide_index=True)
            cols = [c for c in ("n_terms_estimated", "n_terms_after") if c in cluster_path.columns]
            if cols:
                st.line_chart(cluster_path.set_index("iteration")[cols], width="stretch")

    st.markdown("**最后一轮 Readout Top Terms / Final Readout Top Terms**")
    if readout_top_terms.empty:
        st.info("trace payload 里还没有 readout top terms / No readout top terms in the trace payload yet.")
    else:
        st.dataframe(readout_top_terms, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("事件流 / Event Stream")
        st.dataframe(events, width="stretch", hide_index=True)
    with c2:
        st.subheader("指标流 / Metric Stream")
        st.dataframe(metrics, width="stretch", hide_index=True)

    if not metrics.empty:
        st.subheader("指标曲线 / Metric Curves")
        chart_df = _to_chart_frame(metrics)
        if chart_df.empty:
            st.info("当前指标数据还不足以生成曲线 / Not enough metric data to plot curves.")
        else:
            st.line_chart(chart_df, width="stretch")

    st.subheader("训练轨迹（派生 / 诊断 / 路径）/ Training Trace")
    if trace.empty:
        st.info("当前 run 还没有 training trace；如果这是 symbolic 训练，请确认 search_trace 已开启 / No training trace yet.")
    else:
        st.dataframe(derived, width="stretch", hide_index=True)

        curve = _to_trace_curve_frame(trace)
        if not curve.empty:
            st.subheader("训练轨迹曲线 / Training Trace Curves")
            st.line_chart(curve, width="stretch")

        c_diag1, c_diag2 = st.columns(2)
        with c_diag1:
            st.subheader("Hard-Fit 诊断 / Hard-Fit Diagnostics")
            cols = [
                c
                for c in (
                    "iteration",
                    "delta_weight_l2",
                    "delta_grad_mismatch",
                    "delta_effective_rmse",
                    "hard_fit_index",
                )
                if c in derived.columns
            ]
            st.dataframe(derived[cols], width="stretch", hide_index=True)
        with c_diag2:
            st.subheader("Hard-Fit 曲线 / Hard-Fit Curves")
            hf_cols = [c for c in ("hard_fit_index", "delta_effective_rmse") if c in derived.columns]
            if hf_cols:
                st.line_chart(derived.set_index("iteration")[hf_cols], width="stretch")
            else:
                st.info("当前没有可画的 hard-fit 曲线 / No hard-fit curves to draw.")

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("函数族路径 / Family Path")
            fam = trace["selected_family"].fillna("N/A").value_counts().rename_axis("family").reset_index(name="count")
            st.dataframe(fam, width="stretch", hide_index=True)
        with c4:
            st.subheader("操作路径 / Operation Path")
            op = trace["operation"].fillna("N/A").value_counts().rename_axis("operation").reset_index(name="count")
            st.dataframe(op, width="stretch", hide_index=True)

        c5, c6 = st.columns(2)
        with c5:
            st.subheader("替换归因 / Replace Attribution")
            rep = _summarize_replace_attribution(derived)
            if rep.empty:
                st.info("当前 run 没有 replace 归因数据 / No replace attribution data.")
            else:
                st.dataframe(rep, width="stretch", hide_index=True)
        with c6:
            st.subheader("操作切换质量 / Transition Quality")
            trans = _summarize_operation_transition(derived)
            if trans.empty:
                st.info("当前没有可展示的 transition 质量统计 / No transition quality statistics yet.")
            else:
                st.dataframe(trans, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()



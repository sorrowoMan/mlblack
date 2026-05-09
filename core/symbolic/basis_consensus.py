from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from core.symbolic.truth_contracts import normalized_text, term_row_view

_JOINT_CORE_DEFAULT_WEIGHTS: dict[str, float] = {
    "support_rate": 0.50,
    "exact_stability": 0.30,
    "support_weight_rate": 0.20,
    "cross_lane_stability": 0.0,
}
_JOINT_CORE_MULTI_LANE_WEIGHTS: dict[str, float] = {
    "support_rate": 0.40,
    "exact_stability": 0.25,
    "support_weight_rate": 0.15,
    "cross_lane_stability": 0.20,
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(raw) for key, raw in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


def _expr_key(expr: Mapping[str, Any] | None, *, fallback: str) -> str:
    if isinstance(expr, Mapping) and dict(expr):
        return json.dumps(_jsonable(dict(expr)), ensure_ascii=False, sort_keys=True)
    return normalized_text(fallback)


def _text_key(value: Any) -> str | None:
    text = normalized_text(value)
    return text or None


def _lane_inventory(
    runs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    lane_ids = sorted(
        {
            str(value)
            for value in (_text_key(dict(run).get("lane_id")) for run in tuple(runs))
            if value
        }
    )
    lane_families = sorted(
        {
            str(value)
            for value in (_text_key(dict(run).get("lane_family")) for run in tuple(runs))
            if value
        }
    )
    return tuple(lane_ids), tuple(lane_families)


def _joint_core_score_weights(
    *,
    total_lane_count: int,
    total_lane_family_count: int,
) -> dict[str, float]:
    if int(total_lane_count) > 1 or int(total_lane_family_count) > 1:
        return dict(_JOINT_CORE_MULTI_LANE_WEIGHTS)
    return dict(_JOINT_CORE_DEFAULT_WEIGHTS)


def _is_periodic(row: Mapping[str, Any]) -> bool:
    expr = str(row.get("expr") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    return bool(
        ("periodic" in semantic_family)
        or ("sin(" in expr)
        or ("cos(" in expr)
        or ("unary:sin" in semantic_signature)
        or ("unary:cos" in semantic_signature)
    )


def _is_piecewise(row: Mapping[str, Any]) -> bool:
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    return bool(
        row.get("uses_piecewise_gate")
        or ("piecewise" in semantic_family)
        or ("hinge" in name)
        or ("relu(" in expr)
    )


def _is_exp_ratio(row: Mapping[str, Any]) -> bool:
    expr = str(row.get("expr") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    return bool(("exp(" in expr) and ("/" in expr)) or (
        ("unary:exp" in semantic_signature) and ("binary:div" in semantic_signature)
    )


def _is_ratio_like(row: Mapping[str, Any]) -> bool:
    expr = str(row.get("expr") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    return bool(("ratio" in semantic_family) or ("binary:div" in semantic_signature) or ("/" in expr))


def _is_linear_feature(row: Mapping[str, Any]) -> bool:
    expr = str(row.get("expr") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    features = tuple(str(value) for value in tuple(row.get("features", ()) or ()))
    return bool(
        (semantic_family == "linear_feature")
        or semantic_signature.startswith("feature:")
        or (len(features) == 1 and expr == normalized_text(features[0]))
    )


def _family_contract_like(row: Mapping[str, Any]) -> str:
    features = tuple(str(value) for value in tuple(row.get("features", ()) or ()))
    feature_text = ",".join(features)
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    if _is_exp_ratio(row):
        return f"exp_ratio_family({feature_text})" if feature_text else "exp_ratio_family"
    if _is_ratio_like(row):
        if len(features) >= 3:
            return f"product_ratio_family({feature_text})"
        return f"ratio_family({feature_text})" if feature_text else "ratio_family"
    if _is_periodic(row):
        feature_name = features[0] if features else ""
        return f"periodic_family({feature_name})" if feature_name else "periodic_family"
    if _is_piecewise(row):
        feature_name = features[0] if features else ""
        return f"piecewise_gate_family({feature_name})" if feature_name else "piecewise_gate_family"
    if _is_linear_feature(row):
        feature_name = features[0] if features else ""
        return f"linear_feature_family({feature_name})" if feature_name else "linear_feature_family"
    fallback_family = semantic_family or semantic_signature or "unknown"
    return f"{fallback_family}({feature_text})" if feature_text else fallback_family


def _phase_contract_like(row: Mapping[str, Any], *, strict_key: str) -> str:
    if _is_periodic(row):
        features = tuple(str(value) for value in tuple(row.get("features", ()) or ()))
        feature_name = features[0] if features else ""
        return f"periodic_phase_equivalent({feature_name})" if feature_name else "periodic_phase_equivalent"
    return strict_key


def annotate_basis_entries(
    basis_rows: Sequence[Mapping[str, Any]],
    outer_basis_genome: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    genome_rows = tuple(outer_basis_genome or ())
    annotated: list[dict[str, Any]] = []
    for index, raw_row in enumerate(tuple(basis_rows)):
        row = dict(raw_row)
        genome_term = (
            dict(genome_rows[index])
            if index < len(genome_rows) and isinstance(genome_rows[index], Mapping)
            else {}
        )
        expr = dict(genome_term.get("expr", {})) if isinstance(genome_term.get("expr"), Mapping) else {}
        if "name" not in row and genome_term.get("name") is not None:
            row["name"] = genome_term.get("name")
        view = term_row_view(row)
        strict_key = _expr_key(expr, fallback=str(view.get("expr") or view.get("semantic_signature") or view.get("term_name") or ""))
        family_contract = _family_contract_like(view)
        phase_contract = _phase_contract_like(view, strict_key=strict_key)
        annotated.append(
            {
                "entry_index": int(index),
                "term_name": str(view.get("term_name") or row.get("term_name") or row.get("name") or ""),
                "expression": str(view.get("expression") or row.get("expression") or ""),
                "expr": dict(expr),
                "exact_expr_key": str(strict_key),
                "strict_class_id": str(strict_key),
                "phase_class_id": str(phase_contract),
                "family_class_id": str(family_contract),
                "phase_contract": str(phase_contract),
                "family_contract": str(family_contract),
                "feature_names": [str(value) for value in tuple(view.get("features", ()) or ())],
                "semantic_family": str(view.get("semantic_family") or ""),
                "semantic_signature": str(view.get("semantic_signature") or ""),
                "uses_piecewise_gate": bool(view.get("uses_piecewise_gate")),
            }
        )
    return annotated


def _mode_key(entry: Mapping[str, Any], *, equivalence_mode: str) -> str:
    mode = normalized_text(equivalence_mode)
    if mode == "strict":
        return str(entry.get("strict_class_id") or "")
    if mode in {"phase", "phase_equivalent"}:
        return str(entry.get("phase_class_id") or entry.get("strict_class_id") or "")
    if mode in {"family", "family_level"}:
        return str(entry.get("family_class_id") or entry.get("phase_class_id") or entry.get("strict_class_id") or "")
    raise ValueError("equivalence_mode must be strict | phase | family")


def _resolve_min_support_count(
    *,
    total_runs: int,
    min_support_count: int | None,
    min_support_rate: float,
) -> int:
    explicit = 0 if min_support_count is None else int(max(0, min_support_count))
    rate_required = int(max(1, round(float(min_support_rate) * float(max(1, total_runs)))))
    return max(explicit, rate_required)


def _run_weight(run_payload: Mapping[str, Any], *, run_weight_field: str | None) -> float:
    field_name = str(run_weight_field or "").strip()
    if not field_name:
        return 1.0
    try:
        numeric = float(run_payload.get(field_name))
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(numeric):
        return 1.0
    normalized = field_name.lower()
    if normalized.endswith("rmse") or normalized.endswith("_gap") or ("loss" in normalized) or ("error" in normalized):
        return float(1.0 / (1.0 + max(0.0, numeric)))
    if normalized.endswith("r2"):
        return float(max(0.0, (numeric + 1.0) * 0.5))
    return float(max(0.0, numeric))


def build_core_basis_table(
    *,
    runs: Sequence[Mapping[str, Any]],
    equivalence_mode: str = "family",
    min_support_count: int | None = None,
    min_support_rate: float = 0.5,
    run_weight_field: str | None = None,
) -> list[dict[str, Any]]:
    total_runs = int(len(tuple(runs)))
    total_lane_ids, total_lane_families = _lane_inventory(runs)
    total_lane_count = int(len(total_lane_ids))
    total_lane_family_count = int(len(total_lane_families))
    score_weights = _joint_core_score_weights(
        total_lane_count=total_lane_count,
        total_lane_family_count=total_lane_family_count,
    )
    joint_core_protocol = (
        "support_rate+exact_stability+support_weight_rate+cross_lane_stability"
        if float(score_weights.get("cross_lane_stability", 0.0) or 0.0) > 0.0
        else "support_rate+exact_stability+support_weight_rate"
    )
    total_weight = sum(
        float(_run_weight(run_payload, run_weight_field=run_weight_field))
        for run_payload in tuple(runs)
    )
    if total_weight <= 1e-12:
        total_weight = float(max(1, total_runs))
    required_support = _resolve_min_support_count(
        total_runs=total_runs,
        min_support_count=min_support_count,
        min_support_rate=min_support_rate,
    )
    grouped: dict[str, dict[str, Any]] = {}
    for run_position, run_payload in enumerate(tuple(runs)):
        run_index = int(run_payload.get("run_index", run_position))
        run_id = str(run_payload.get("run_id") or "")
        run_weight = float(_run_weight(run_payload, run_weight_field=run_weight_field))
        lane_id = _text_key(run_payload.get("lane_id"))
        lane_family = _text_key(run_payload.get("lane_family"))
        seen_in_run: set[str] = set()
        for entry in tuple(run_payload.get("basis_entries", ()) or ()):
            if not isinstance(entry, Mapping):
                continue
            class_id = _mode_key(entry, equivalence_mode=equivalence_mode)
            if not class_id:
                continue
            bucket = grouped.setdefault(
                class_id,
                {
                    "equivalence_mode": normalized_text(equivalence_mode),
                    "basis_class_id": class_id,
                    "run_indices": set(),
                    "run_ids": set(),
                    "lane_ids": set(),
                    "lane_families": set(),
                    "support_weight": 0.0,
                    "occurrence_count": 0,
                    "expression_counter": Counter(),
                    "semantic_family_counter": Counter(),
                    "feature_counter": Counter(),
                    "exact_variant_counter": Counter(),
                    "exact_variant_run_indices": defaultdict(set),
                    "representative_entries": {},
                },
            )
            bucket["occurrence_count"] += 1
            bucket["expression_counter"][str(entry.get("expression") or "")] += 1
            bucket["semantic_family_counter"][str(entry.get("semantic_family") or "")] += 1
            bucket["feature_counter"][tuple(str(value) for value in tuple(entry.get("feature_names", ()) or ()))] += 1
            exact_key = str(entry.get("exact_expr_key") or "")
            if exact_key:
                bucket["exact_variant_counter"][exact_key] += 1
                bucket["exact_variant_run_indices"][exact_key].add(run_index)
                bucket["representative_entries"].setdefault(exact_key, dict(entry))
            if class_id not in seen_in_run:
                bucket["run_indices"].add(run_index)
                if run_id:
                    bucket["run_ids"].add(run_id)
                if lane_id:
                    bucket["lane_ids"].add(str(lane_id))
                if lane_family:
                    bucket["lane_families"].add(str(lane_family))
                bucket["support_weight"] += float(run_weight)
                seen_in_run.add(class_id)

    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        run_indices = sorted(int(value) for value in set(bucket["run_indices"]))
        support_count = int(len(run_indices))
        support_rate = float(support_count) / float(max(1, total_runs))
        support_weight = float(bucket.get("support_weight", 0.0) or 0.0)
        support_weight_rate = float(support_weight) / float(max(total_weight, 1e-12))
        lane_ids = sorted(str(value) for value in set(bucket.get("lane_ids", set())) if str(value))
        lane_families = sorted(
            str(value) for value in set(bucket.get("lane_families", set())) if str(value)
        )
        cross_lane_support_count = int(len(lane_ids))
        cross_lane_family_count = int(len(lane_families))
        cross_lane_support_rate = (
            None
            if total_lane_count <= 1
            else float(cross_lane_support_count) / float(max(1, total_lane_count))
        )
        cross_lane_family_support_rate = (
            None
            if total_lane_family_count <= 1
            else float(cross_lane_family_count) / float(max(1, total_lane_family_count))
        )
        expression_counter: Counter[str] = bucket["expression_counter"]
        semantic_family_counter: Counter[str] = bucket["semantic_family_counter"]
        feature_counter: Counter[tuple[str, ...]] = bucket["feature_counter"]
        exact_variant_counter: Counter[str] = bucket["exact_variant_counter"]
        exact_variant_run_indices: Mapping[str, set[int]] = bucket["exact_variant_run_indices"]
        representative_expression = expression_counter.most_common(1)[0][0] if expression_counter else ""
        representative_semantic_family = semantic_family_counter.most_common(1)[0][0] if semantic_family_counter else ""
        representative_features = list(feature_counter.most_common(1)[0][0]) if feature_counter else []
        dominant_exact_key = ""
        dominant_exact_support_count = 0
        dominant_exact_occurrence_count = 0
        if exact_variant_counter:
            dominant_rows: list[tuple[int, int, str]] = []
            for exact_key, occurrence_count in exact_variant_counter.items():
                exact_support_count = int(len(set(exact_variant_run_indices.get(str(exact_key), set()))))
                dominant_rows.append(
                    (
                        int(exact_support_count),
                        int(occurrence_count),
                        str(exact_key),
                    )
                )
            dominant_rows.sort(key=lambda item: (-int(item[0]), -int(item[1]), str(item[2])))
            dominant_exact_support_count, dominant_exact_occurrence_count, dominant_exact_key = dominant_rows[0]
        dominant_exact_support_rate = float(dominant_exact_support_count) / float(max(1, total_runs))
        exact_stability = float(dominant_exact_support_count) / float(max(1, support_count))
        cross_lane_components = [
            float(value)
            for value in (cross_lane_support_rate, cross_lane_family_support_rate)
            if value is not None
        ]
        cross_lane_stability = (
            None
            if not cross_lane_components
            else float(sum(cross_lane_components) / float(len(cross_lane_components)))
        )
        joint_core_score = float(
            float(score_weights["support_rate"]) * float(support_rate)
            + float(score_weights["exact_stability"]) * float(exact_stability)
            + float(score_weights["support_weight_rate"]) * float(support_weight_rate)
            + float(score_weights["cross_lane_stability"]) * float(cross_lane_stability or 0.0)
        )
        selected_as_core = bool(support_count >= required_support and support_rate >= float(min_support_rate))
        rows.append(
            {
                "equivalence_mode": str(bucket["equivalence_mode"]),
                "basis_class_id": str(bucket["basis_class_id"]),
                "representative_expression": str(representative_expression),
                "representative_semantic_family": str(representative_semantic_family),
                "feature_names": representative_features,
                "support_count": int(support_count),
                "support_rate": float(support_rate),
                "support_weight": float(support_weight),
                "support_weight_rate": float(support_weight_rate),
                "occurrence_count": int(bucket["occurrence_count"]),
                "run_indices": run_indices,
                "run_ids": sorted(str(value) for value in set(bucket["run_ids"]) if str(value)),
                "lane_ids": lane_ids,
                "lane_families": lane_families,
                "exact_variant_count": int(len(exact_variant_counter)),
                "dominant_exact_expr_key": str(dominant_exact_key),
                "dominant_exact_support_count": int(dominant_exact_support_count),
                "dominant_exact_support_rate": float(dominant_exact_support_rate),
                "dominant_exact_occurrence_count": int(dominant_exact_occurrence_count),
                "exact_stability": float(exact_stability),
                "multi_run_core_frequency": float(support_rate),
                "cross_lane_support_count": int(cross_lane_support_count),
                "cross_lane_support_rate": cross_lane_support_rate,
                "cross_lane_family_count": int(cross_lane_family_count),
                "cross_lane_family_support_rate": cross_lane_family_support_rate,
                "cross_lane_stability": cross_lane_stability,
                "joint_core_score": float(joint_core_score),
                "joint_core_score_protocol": str(joint_core_protocol),
                "selected_as_core": bool(selected_as_core),
                "required_support_count": int(required_support),
                "min_support_rate": float(min_support_rate),
            }
        )
    rows.sort(
        key=lambda item: (
            not bool(item.get("selected_as_core")),
            -float(item.get("joint_core_score", 0.0)),
            -int(item.get("support_count", 0)),
            -float(item.get("support_rate", 0.0)),
            -float(item.get("support_weight_rate", 0.0)),
            -int(item.get("occurrence_count", 0)),
            str(item.get("basis_class_id") or ""),
        )
    )
    return rows


def build_core_basis_tables(
    *,
    runs: Sequence[Mapping[str, Any]],
    equivalence_modes: Sequence[str] = ("strict", "phase", "family"),
    min_support_count: int | None = None,
    min_support_rate: float = 0.5,
    run_weight_field: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(mode): build_core_basis_table(
            runs=runs,
            equivalence_mode=str(mode),
            min_support_count=min_support_count,
            min_support_rate=min_support_rate,
            run_weight_field=run_weight_field,
        )
        for mode in tuple(equivalence_modes)
    }


def select_locked_core_seed_genome(
    *,
    runs: Sequence[Mapping[str, Any]],
    equivalence_mode: str = "family",
    min_support_count: int | None = None,
    min_support_rate: float = 0.5,
    max_terms: int | None = None,
    min_seed_terms: int = 0,
    backfill_mode: str = "none",
    run_weight_field: str | None = None,
) -> dict[str, Any]:
    total_lane_ids, total_lane_families = _lane_inventory(runs)
    score_weights = _joint_core_score_weights(
        total_lane_count=len(total_lane_ids),
        total_lane_family_count=len(total_lane_families),
    )
    table = build_core_basis_table(
        runs=runs,
        equivalence_mode=equivalence_mode,
        min_support_count=min_support_count,
        min_support_rate=min_support_rate,
        run_weight_field=run_weight_field,
    )
    raw_selected_rows = [dict(row) for row in table if bool(row.get("selected_as_core"))]
    selected_rows = list(raw_selected_rows)
    max_row_cap = None if max_terms is None else int(max(0, max_terms))
    if max_row_cap is not None:
        selected_rows = selected_rows[:max_row_cap]
    target_seed_terms = int(max(0, min_seed_terms))
    if str(backfill_mode or "none").strip().lower() == "weighted_rank" and target_seed_terms > len(selected_rows):
        selected_ids = {str(row.get("basis_class_id") or "") for row in selected_rows}
        ranked_backfill = [
            dict(row)
            for row in table
            if str(row.get("basis_class_id") or "") and str(row.get("basis_class_id") or "") not in selected_ids
        ]
        ranked_backfill.sort(
            key=lambda item: (
                -float(item.get("joint_core_score", 0.0) or 0.0),
                -float(item.get("support_weight_rate", item.get("support_rate", 0.0)) or 0.0),
                -int(item.get("support_count", 0)),
                -float(item.get("support_rate", 0.0)),
                -int(item.get("occurrence_count", 0)),
                str(item.get("basis_class_id") or ""),
            )
        )
        add_count = int(target_seed_terms - len(selected_rows))
        if max_row_cap is not None:
            add_count = min(add_count, max(0, max_row_cap - len(selected_rows)))
        if add_count > 0:
            selected_rows.extend(ranked_backfill[:add_count])
    if max_row_cap is not None:
        selected_rows = selected_rows[:max_row_cap]
    selected_class_ids = {str(row.get("basis_class_id") or "") for row in selected_rows}
    raw_selected_ids = {str(row.get("basis_class_id") or "") for row in raw_selected_rows}
    exact_variant_support: dict[str, dict[str, Any]] = {}
    class_to_exact_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_position, run_payload in enumerate(tuple(runs)):
        run_index = int(run_payload.get("run_index", run_position))
        for entry in tuple(run_payload.get("basis_entries", ()) or ()):
            if not isinstance(entry, Mapping):
                continue
            class_id = _mode_key(entry, equivalence_mode=equivalence_mode)
            if class_id not in selected_class_ids:
                continue
            exact_key = str(entry.get("exact_expr_key") or "")
            if not exact_key:
                continue
            bucket = exact_variant_support.setdefault(
                exact_key,
                {
                    "entry": dict(entry),
                    "run_indices": set(),
                    "occurrence_count": 0,
                },
            )
            bucket["run_indices"].add(run_index)
            bucket["occurrence_count"] += 1
            class_to_exact_rows[class_id].append(dict(entry))

    seed_genome: list[dict[str, Any]] = []
    selected_core_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        class_id = str(row.get("basis_class_id") or "")
        candidates = class_to_exact_rows.get(class_id, [])
        if not candidates:
            continue
        candidate_support: list[tuple[int, int, str, dict[str, Any]]] = []
        seen_exact: set[str] = set()
        for entry in candidates:
            exact_key = str(entry.get("exact_expr_key") or "")
            if not exact_key or exact_key in seen_exact:
                continue
            seen_exact.add(exact_key)
            support = exact_variant_support.get(exact_key, {})
            candidate_support.append(
                (
                    int(len(set(support.get("run_indices", set())))),
                    int(support.get("occurrence_count", 0)),
                    exact_key,
                    dict(entry),
                )
            )
        if not candidate_support:
            continue
        candidate_support.sort(key=lambda item: (-int(item[0]), -int(item[1]), str(item[2])))
        _, exact_occurrence_count, exact_key, chosen_entry = candidate_support[0]
        expr = dict(chosen_entry.get("expr", {}))
        if not expr:
            continue
        seed_genome.append(
            {
                "name": str(chosen_entry.get("term_name") or chosen_entry.get("expression") or f"core_basis_{len(seed_genome)}"),
                "expr": expr,
            }
        )
        selected_core_rows.append(
            {
                **dict(row),
                "representative_exact_expr_key": str(exact_key),
                "representative_exact_occurrence_count": int(exact_occurrence_count),
                "representative_exact_support_count": int(len(set(exact_variant_support.get(exact_key, {}).get("run_indices", set())))),
                "representative_exact_support_rate": float(
                    int(len(set(exact_variant_support.get(exact_key, {}).get("run_indices", set()))))
                    / float(max(1, len(tuple(runs))))
                ),
                "representative_seed_name": str(chosen_entry.get("term_name") or ""),
                "representative_seed_expression": str(chosen_entry.get("expression") or ""),
                "representative_phase_class_id": str(chosen_entry.get("phase_class_id") or ""),
                "representative_family_class_id": str(chosen_entry.get("family_class_id") or ""),
                "representative_strict_class_id": str(chosen_entry.get("strict_class_id") or chosen_entry.get("exact_expr_key") or ""),
                "representative_semantic_family": str(chosen_entry.get("semantic_family") or ""),
                "representative_feature_names": [
                    str(value) for value in tuple(chosen_entry.get("feature_names", ()) or ())
                ],
                "selection_source": (
                    "consensus"
                    if class_id in raw_selected_ids
                    else "weighted_backfill"
                ),
            }
        )
    return {
        "equivalence_mode": normalized_text(equivalence_mode),
        "core_basis_table": table,
        "selected_core_rows": selected_core_rows,
        "seed_genome": tuple(seed_genome),
        "selection_strategy": {
            "backfill_mode": str(backfill_mode or "none").strip().lower() or "none",
            "min_seed_terms": int(target_seed_terms),
            "run_weight_field": str(run_weight_field or ""),
            "joint_core_score_weights": {str(key): float(value) for key, value in score_weights.items()},
            "joint_core_score_protocol": (
                "support_rate+exact_stability+support_weight_rate+cross_lane_stability"
                if float(score_weights.get("cross_lane_stability", 0.0) or 0.0) > 0.0
                else "support_rate+exact_stability+support_weight_rate"
            ),
            "lane_count": int(len(total_lane_ids)),
            "lane_family_count": int(len(total_lane_families)),
        },
    }


__all__ = [
    "annotate_basis_entries",
    "build_core_basis_table",
    "build_core_basis_tables",
    "select_locked_core_seed_genome",
]

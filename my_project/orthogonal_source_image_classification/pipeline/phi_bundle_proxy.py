from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.orthogonal_source import OrthogonalSourceConfig, OrthogonalSourceLayer
from my_project.orthogonal_source_image_classification.config import ImageClassificationConfig
from my_project.orthogonal_source_image_classification.problem import load_image_classification_dataset
from my_project.orthogonal_source_image_classification.pipeline.representation_search import (
    RepresentationFormulaSearchResult,
    _build_formula_pool,
    _one_vs_rest_target,
    _safe_corr,
    _target_corr_score,
)


@dataclass(frozen=True)
class PhiBundleEvaluationConfig:
    dataset_key: str = "digits"
    train_ratio: float = 0.75
    seed: int = 42
    max_rows: int = 900
    fallback_objective: float = 10.0


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(raw) for key, raw in dict(value).items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [dict(row) for row in rows]
    if not items:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in items:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)


def _enabled_lanes(bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    lanes = bundle.get("lanes", ())
    out: list[Mapping[str, Any]] = []
    for lane in tuple(lanes or ()):
        if not isinstance(lane, Mapping):
            continue
        if not bool(lane.get("enabled", True)):
            continue
        family = str(lane.get("family", "")).strip()
        if family:
            out.append(dict(lane))
    return tuple(out)


def _enabled_families(bundle: Mapping[str, Any]) -> tuple[str, ...]:
    families: list[str] = []
    for lane in _enabled_lanes(bundle):
        family = str(lane.get("family", "")).strip()
        if family and family not in families:
            families.append(family)
    return tuple(families)


def _bucket(value: Any, labels: Sequence[str], *, default: str) -> str:
    if not labels:
        return str(default)
    try:
        raw = float(value)
    except Exception:
        return str(default)
    idx = int(np.clip(np.floor(raw * len(labels)), 0, len(labels) - 1))
    return str(labels[idx])


def _extract_dct_uv(name: str) -> tuple[int, int] | None:
    raw = str(name)
    if not raw.startswith("dct_u") or "_v" not in raw:
        return None
    try:
        left, right = raw[5:].split("_v", 1)
        return int(left), int(right)
    except Exception:
        return None


def _extract_patch_meta(name: str) -> dict[str, int | str] | None:
    raw = str(name)
    if not raw.startswith("patch") or "x" not in raw:
        return None
    try:
        size_raw, rest = raw[5:].split("_", 1)
        size = int(size_raw.split("x", 1)[0])
        parts = rest.split("_")
        if parts[0].startswith("s"):
            stride = int(parts[0][1:])
            row = int(parts[1][1:])
            col = int(parts[2][1:])
            op = str(parts[3])
            return {"size": size, "stride": stride, "row": row, "col": col, "op": op}
        if parts[0].startswith("rpx"):
            row = int(parts[0][3:])
            col = int(parts[1][3:])
            op = str(parts[2])
            return {"size": size, "stride": 1, "row": row, "col": col, "op": op}
        row_cell = int(parts[0][1:])
        col_cell = int(parts[1][1:])
        op = "sum" if str(parts[2]) == "ink" else str(parts[2])
        return {"size": size, "stride": size, "row": row_cell * size, "col": col_cell * size, "op": op}
    except Exception:
        return None


def _patch_region_allowed(meta: Mapping[str, int | str], region: str) -> bool:
    if region == "all":
        return True
    size = int(meta.get("size", 0))
    row = int(meta.get("row", 0))
    col = int(meta.get("col", 0))
    row_center = row + (size / 2.0)
    col_center = col + (size / 2.0)
    if region == "center":
        return 2.0 <= row_center <= 6.0 and 2.0 <= col_center <= 6.0
    if region == "corner":
        return row in {0, 8 - size} and col in {0, 8 - size}
    if region == "outer":
        return row == 0 or col == 0 or row + size >= 8 or col + size >= 8
    return True


def _lane_allows_feature(*, lane: Mapping[str, Any], family: str, name: str) -> bool:
    fam = str(family)
    raw_name = str(name)
    param = lane.get("param", 1.0)
    if fam == "edge":
        if "horizontal_edge" in raw_name:
            direction = "horizontal"
        elif "vertical_edge" in raw_name:
            direction = "vertical"
        else:
            return False
        scope = "global" if raw_name.startswith("global_") else "local"
        if raw_name.endswith("_signed"):
            operator = "signed"
        elif raw_name.endswith("_squared"):
            operator = "squared"
        else:
            operator = "abs"
        wanted_direction = str(lane.get("edge_direction", "")).strip()
        wanted_scope = str(lane.get("edge_scope", "")).strip()
        wanted_operator = str(lane.get("edge_operator", "")).strip()
        if not wanted_direction or not wanted_scope or not wanted_operator:
            mode = str(lane.get("edge_mode") or _bucket(param, ("horizontal", "vertical", "global", "all"), default="all"))
            if mode == "horizontal":
                wanted_direction, wanted_scope, wanted_operator = "horizontal", "all", "abs"
            elif mode == "vertical":
                wanted_direction, wanted_scope, wanted_operator = "vertical", "all", "abs"
            elif mode == "global":
                wanted_direction, wanted_scope, wanted_operator = "both", "global", "abs"
            else:
                wanted_direction, wanted_scope, wanted_operator = "both", "all", "all"
        direction_ok = wanted_direction in {"both", "all"} or direction == wanted_direction
        scope_ok = wanted_scope == "all" or scope == wanted_scope
        operator_ok = wanted_operator == "all" or operator == wanted_operator
        return bool(direction_ok and scope_ok and operator_ok)
    if fam in {"patch_pool", "patch_texture"}:
        meta = _extract_patch_meta(raw_name)
        if meta is None:
            return False
        patch_size = int(lane.get("patch_size") or (2 if float(param) < 0.5 else 4))
        if int(meta.get("size", 0)) != patch_size:
            return False
        patch_stride = lane.get("patch_stride", "all")
        if str(patch_stride) != "all":
            stride = int(patch_stride)
            if int(meta.get("row", 0)) % stride != 0 or int(meta.get("col", 0)) % stride != 0:
                return False
        region = str(lane.get("patch_region", "all"))
        if not _patch_region_allowed(meta, region):
            return False
        op = str(meta.get("op", ""))
        if fam == "patch_pool":
            pooling = str(lane.get("patch_pooling", "all"))
            return pooling == "all" or op == pooling
        texture_operator = str(lane.get("texture_operator", "all"))
        return texture_operator == "all" or op == texture_operator
    if fam == "orthogonal_frequency":
        band = str(lane.get("dct_band") or _bucket(param, ("low", "mid", "high", "all"), default="all"))
        uv = _extract_dct_uv(raw_name)
        if uv is None:
            return False
        freq = int(uv[0] + uv[1])
        if band == "low":
            band_ok = freq <= 2
        elif band == "mid":
            band_ok = 2 <= freq <= 4
        elif band == "high":
            band_ok = freq >= 4
        else:
            band_ok = True
        orientation = str(lane.get("dct_orientation", "all"))
        if orientation == "row":
            orientation_ok = uv[0] > uv[1]
        elif orientation == "col":
            orientation_ok = uv[1] > uv[0]
        elif orientation == "diagonal":
            orientation_ok = uv[0] == uv[1]
        else:
            orientation_ok = True
        return bool(band_ok and orientation_ok)
    if fam == "moment":
        axis = str(lane.get("moment_axis", "")).strip()
        stat = str(lane.get("moment_stat", "")).strip()
        if not axis or not stat:
            mode = str(lane.get("moment_mode") or _bucket(param, ("center", "variance", "row", "col", "all"), default="all"))
            if mode == "center":
                axis, stat = "both", "center"
            elif mode == "variance":
                axis, stat = "both", "variance"
            elif mode == "row":
                axis, stat = "row", "all"
            elif mode == "col":
                axis, stat = "col", "all"
            else:
                axis, stat = "both", "all"
        name_axis = "row" if raw_name.startswith("row_") else "col" if raw_name.startswith("col_") else ""
        name_stat = "center" if raw_name.endswith("_center") else "variance" if raw_name.endswith("_variance") else ""
        axis_ok = axis in {"both", "all"} or axis == name_axis
        stat_ok = stat == "all" or stat == name_stat
        return bool(axis_ok and stat_ok)
    if fam == "region":
        mode = str(lane.get("region_mode") or _bucket(param, ("center", "outer_ring", "all"), default="all"))
        if mode == "center":
            return raw_name == "center_4x4_ink"
        if mode == "outer_ring":
            return raw_name == "outer_ring_ink"
        return True
    if fam == "symmetry":
        axis = str(lane.get("symmetry_axis") or _bucket(param, ("left_right", "top_bottom", "all"), default="all"))
        if axis == "left_right":
            return raw_name == "left_right_symmetry_error"
        if axis == "top_bottom":
            return raw_name == "top_bottom_symmetry_error"
        return True
    if fam == "row_projection":
        band = str(lane.get("row_band") or _bucket(param, ("top", "middle", "bottom", "all"), default="all"))
        if not raw_name.startswith("row_") or not raw_name.endswith("_ink"):
            return False
        try:
            row_idx = int(raw_name.split("_")[1])
        except Exception:
            return False
        if band == "top":
            return row_idx <= 2
        if band == "middle":
            return 2 <= row_idx <= 5
        if band == "bottom":
            return row_idx >= 5
        return True
    if fam == "col_projection":
        band = str(lane.get("col_band") or _bucket(param, ("left", "middle", "right", "all"), default="all"))
        if not raw_name.startswith("col_") or not raw_name.endswith("_ink"):
            return False
        try:
            col_idx = int(raw_name.split("_")[1])
        except Exception:
            return False
        if band == "left":
            return col_idx <= 2
        if band == "middle":
            return 2 <= col_idx <= 5
        if band == "right":
            return col_idx >= 5
        return True
    if fam == "mass":
        return raw_name == "total_ink"
    return True


def _clip_int(value: Any, *, low: int, high: int, default: int) -> int:
    try:
        raw = int(round(float(value)))
    except Exception:
        raw = int(default)
    return int(np.clip(raw, int(low), int(high)))


def _clip_float(value: Any, *, low: float, high: float, default: float) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = float(default)
    return float(np.clip(raw, float(low), float(high)))


def _select_representation_from_bundle(
    *,
    dataset,
    bundle: Mapping[str, Any],
) -> RepresentationFormulaSearchResult:
    pool_train, pool_test, names, expressions, families = _build_formula_pool(dataset.X_train, dataset.X_test)
    lanes = _enabled_lanes(bundle)
    if not lanes:
        lanes = (
            {"family": "edge", "edge_mode": "all", "param": 1.0},
            {"family": "patch_pool", "patch_size": 2, "param": 0.0},
            {"family": "orthogonal_frequency", "dct_band": "all", "param": 1.0},
        )
    allowed = {str(lane.get("family", "")).strip() for lane in lanes if str(lane.get("family", "")).strip()}
    target = _one_vs_rest_target(dataset.y_train)
    scores = np.asarray([_target_corr_score(pool_train[:, j], target) for j in range(pool_train.shape[1])], dtype=float)
    family_arr = np.asarray(families, dtype=object)
    allowed_indices = np.asarray(
        [
            idx
            for idx, family in enumerate(family_arr)
            for lane in lanes
            if str(family) == str(lane.get("family", "")).strip()
            and _lane_allows_feature(lane=lane, family=str(family), name=str(names[idx]))
        ],
        dtype=int,
    )
    if allowed_indices.size > 0:
        allowed_indices = np.unique(allowed_indices)
    if allowed_indices.size == 0:
        fallback_allowed = set(allowed) if allowed else {"edge", "patch_pool", "orthogonal_frequency"}
        allowed_indices = np.asarray([idx for idx, family in enumerate(family_arr) if str(family) in fallback_allowed], dtype=int)
    if allowed_indices.size == 0:
        allowed_indices = np.arange(pool_train.shape[1], dtype=int)

    max_features = _clip_int(bundle.get("representation_max_features"), low=4, high=96, default=32)
    keep_top = _clip_int(bundle.get("representation_candidate_keep_top"), low=4, high=160, default=80)
    max_abs_corr = _clip_float(bundle.get("representation_max_pair_abs_corr"), low=0.65, high=0.999, default=0.96)
    ranked_allowed = allowed_indices[np.argsort(-scores[allowed_indices])]
    screened = tuple(int(i) for i in ranked_allowed[: min(keep_top, ranked_allowed.size)])
    selected: list[int] = []
    for idx in screened:
        if len(selected) >= max_features:
            break
        if not selected:
            selected.append(idx)
            continue
        corr_max = max(abs(_safe_corr(pool_train[:, idx], pool_train[:, prev])) for prev in selected)
        if corr_max <= max_abs_corr:
            selected.append(idx)
    if not selected:
        selected = [int(screened[0])] if screened else [int(np.argmax(scores))]

    selected_tuple = tuple(int(i) for i in selected)
    selected_set = set(selected_tuple)
    selected_rank = {idx: rank for rank, idx in enumerate(selected_tuple)}
    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(screened):
        rows.append(
            {
                "pool_rank": int(rank),
                "selected_rank": "" if idx not in selected_set else int(selected_rank[idx]),
                "selected": bool(idx in selected_set),
                "name": str(names[idx]),
                "expression": str(expressions[idx]),
                "family": str(families[idx]),
                "target_corr": float(scores[idx]),
            }
        )
    return RepresentationFormulaSearchResult(
        pool_train=pool_train[:, screened] if screened else np.empty((pool_train.shape[0], 0), dtype=float),
        pool_test=pool_test[:, screened] if screened else np.empty((pool_test.shape[0], 0), dtype=float),
        selected_train=pool_train[:, selected_tuple],
        selected_test=pool_test[:, selected_tuple],
        pool_feature_names=tuple(names[i] for i in screened),
        selected_feature_names=tuple(names[i] for i in selected_tuple),
        formula_rows=tuple(rows),
        report={
            "component": "phi_bundle_formula_selection",
            "allowed_families": tuple(sorted(allowed)),
            "allowed_lanes": tuple(dict(lane) for lane in lanes),
            "candidate_count": int(pool_train.shape[1]),
            "screened_count": int(len(screened)),
            "selected_count": int(len(selected_tuple)),
            "selection_policy": "outer_bundle_family_filter_then_target_corr_with_pairwise_redundancy_cap",
            "max_pair_abs_corr": float(max_abs_corr),
        },
    )


def _fit_logistic_metrics(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, float]:
    if np.asarray(X_train).shape[1] <= 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=800, C=1.0, solver="lbfgs", random_state=int(seed)),
    )
    model.fit(np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=int).reshape(-1))
    pred = model.predict(np.asarray(X_test, dtype=float))
    y = np.asarray(y_test, dtype=int).reshape(-1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }


def _run_identity_orthogonal_sources(
    *,
    representation: RepresentationFormulaSearchResult,
    y_train: np.ndarray,
    max_sources: int,
    candidate_keep_top: int,
    max_pair_abs_corr: float,
    metadata: Mapping[str, Any],
):
    layer = OrthogonalSourceLayer(
        OrthogonalSourceConfig(
            max_sources=int(max_sources),
            candidate_keep_top=int(candidate_keep_top),
            max_pair_abs_corr=float(max_pair_abs_corr),
            target_task="classification",
            min_abs_target_corr=0.015,
            include_raw=True,
            include_unary=False,
            include_pairwise=False,
            include_triple_ratio=False,
            include_hinge=False,
            include_exp_ratio=False,
        )
    )
    return layer.fit_transform(
        X_train=representation.selected_train,
        y_train=np.asarray(y_train, dtype=int).reshape(-1),
        X_test=representation.selected_test,
        feature_names=representation.selected_feature_names,
        metadata=dict(metadata),
    )


def evaluate_phi_bundle(
    bundle: Mapping[str, Any],
    *,
    config: PhiBundleEvaluationConfig | None = None,
    output_dir: str | Path | None = None,
    run_label: str = "",
) -> dict[str, Any]:
    cfg = config or PhiBundleEvaluationConfig()
    try:
        dataset = load_image_classification_dataset(
            dataset_key=str(cfg.dataset_key),
            train_ratio=float(cfg.train_ratio),
            max_rows=int(cfg.max_rows),
            seed=int(cfg.seed),
        )
        representation = _select_representation_from_bundle(dataset=dataset, bundle=bundle)
        max_sources = _clip_int(bundle.get("max_sources"), low=2, high=64, default=16)
        candidate_keep_top = _clip_int(bundle.get("orth_candidate_keep_top"), low=2, high=160, default=max_sources)
        max_pair_abs_corr = _clip_float(bundle.get("orth_max_pair_abs_corr"), low=0.4, high=0.98, default=0.76)
        source_result = _run_identity_orthogonal_sources(
            representation=representation,
            y_train=dataset.y_train,
            max_sources=max_sources,
            candidate_keep_top=candidate_keep_top,
            max_pair_abs_corr=max_pair_abs_corr,
            metadata={
                "target_task": "classification",
                "raw_observation_space": "flattened_8x8_pixels",
                "phi_bundle": dict(bundle),
                "representation_report": dict(representation.report),
            },
        )
        selected_metrics = _fit_logistic_metrics(
            X_train=representation.selected_train,
            X_test=representation.selected_test,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            seed=int(cfg.seed),
        )
        orth_metrics = _fit_logistic_metrics(
            X_train=source_result.train_basis,
            X_test=source_result.test_basis,
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            seed=int(cfg.seed),
        )
        augmented_metrics = _fit_logistic_metrics(
            X_train=np.hstack([representation.selected_train, source_result.train_basis]),
            X_test=np.hstack([representation.selected_test, source_result.test_basis]),
            y_train=dataset.y_train,
            y_test=dataset.y_test,
            seed=int(cfg.seed),
        )
        best_accuracy = float(
            max(
                selected_metrics["accuracy"],
                orth_metrics["accuracy"],
                augmented_metrics["accuracy"],
            )
        )
        lane_count = int(len(_enabled_families(bundle)))
        selected_count = int(representation.report.get("selected_count", representation.selected_train.shape[1]))
        source_count = int(source_result.report.get("selected_source_count", source_result.train_basis.shape[1]))
        generated_count = int(representation.report.get("screened_count", representation.pool_train.shape[1]))
        redundancy = float(source_result.report.get("pair_abs_corr_mean", 0.0))
        instability = float(1.0 - float(source_result.report.get("mean_source_stability", 0.0)))
        complexity = float(
            (0.34 * min(1.0, lane_count / 10.0))
            + (0.33 * min(1.0, selected_count / 80.0))
            + (0.33 * min(1.0, source_count / 32.0))
        )
        cost = float(min(1.0, generated_count / 120.0))
        objectives = [float(1.0 - best_accuracy), redundancy, complexity, max(0.0, instability), cost]
        metrics = {
            "selected_accuracy": float(selected_metrics["accuracy"]),
            "orthogonal_accuracy": float(orth_metrics["accuracy"]),
            "augmented_accuracy": float(augmented_metrics["accuracy"]),
            "best_accuracy": best_accuracy,
            "best_feature_space": max(
                (
                    ("image_representation", selected_metrics["accuracy"]),
                    ("orthogonal_sources", orth_metrics["accuracy"]),
                    ("image_representation_plus_orthogonal_sources", augmented_metrics["accuracy"]),
                ),
                key=lambda item: item[1],
            )[0],
            "selected_macro_f1": float(selected_metrics["macro_f1"]),
            "orthogonal_macro_f1": float(orth_metrics["macro_f1"]),
            "augmented_macro_f1": float(augmented_metrics["macro_f1"]),
        }
        result = {
            "status": "ok",
            "run_label": str(run_label),
            "bundle": dict(bundle),
            "objectives": objectives,
            "metrics": metrics,
            "representation_report": dict(representation.report),
            "source_report": dict(source_result.report),
            "representation_formula_rows": tuple(representation.formula_rows),
            "orthogonal_source_rows": tuple(source_result.source_rows),
        }
    except Exception as exc:
        fallback = float(cfg.fallback_objective)
        result = {
            "status": "error",
            "run_label": str(run_label),
            "bundle": dict(bundle),
            "objectives": [fallback, fallback, fallback, fallback, fallback],
            "metrics": {"error": f"{type(exc).__name__}: {exc}"},
            "representation_report": {},
            "source_report": {},
            "representation_formula_rows": (),
            "orthogonal_source_rows": (),
        }

    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "phi_bundle_evaluation.json").write_text(
            json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_csv(out / "representation_formula_table.csv", result.get("representation_formula_rows", ()))
        _write_csv(out / "orthogonal_source_table.csv", result.get("orthogonal_source_rows", ()))
    return result


__all__ = ["PhiBundleEvaluationConfig", "evaluate_phi_bundle"]

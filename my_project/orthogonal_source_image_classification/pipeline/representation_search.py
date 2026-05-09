from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from my_project.orthogonal_source_image_classification.config import ImageClassificationConfig
from my_project.orthogonal_source_image_classification.problem import ImageClassificationDataset


@dataclass(frozen=True)
class RepresentationFormulaSearchResult:
    pool_train: np.ndarray
    pool_test: np.ndarray
    selected_train: np.ndarray
    selected_test: np.ndarray
    pool_feature_names: tuple[str, ...]
    selected_feature_names: tuple[str, ...]
    formula_rows: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    if x.size == 0 or y.size == 0 or x.size != y.size:
        return 0.0
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y)))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _one_vs_rest_target(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int).reshape(-1)
    classes = tuple(int(v) for v in np.unique(labels))
    out = np.zeros((labels.size, len(classes)), dtype=float)
    for j, cls in enumerate(classes):
        out[:, j] = (labels == cls).astype(float)
    return out


def _target_corr_score(values: np.ndarray, target_matrix: np.ndarray) -> float:
    col = np.asarray(values, dtype=float).reshape(-1)
    if target_matrix.ndim != 2 or target_matrix.shape[0] != col.size:
        return 0.0
    return float(max(abs(_safe_corr(col, target_matrix[:, j])) for j in range(target_matrix.shape[1])))


def _append_feature(
    features: list[np.ndarray],
    names: list[str],
    expressions: list[str],
    families: list[str],
    train_col: np.ndarray,
    test_col: np.ndarray,
    *,
    name: str,
    expression: str,
    family: str,
) -> None:
    tr = np.asarray(train_col, dtype=float).reshape(-1, 1)
    te = np.asarray(test_col, dtype=float).reshape(-1, 1)
    if tr.shape[0] <= 0 or te.shape[0] <= 0:
        return
    if float(np.nanstd(tr)) <= 1.0e-12:
        return
    features.append((tr, te))
    names.append(str(name))
    expressions.append(str(expression))
    families.append(str(family))


def _dct_basis(images: np.ndarray, u: int, v: int) -> np.ndarray:
    arr = np.asarray(images, dtype=float).reshape(-1, 8, 8)
    xs = np.arange(8, dtype=float).reshape(1, 8, 1)
    ys = np.arange(8, dtype=float).reshape(1, 1, 8)
    basis = np.cos((2.0 * xs + 1.0) * float(u) * np.pi / 16.0) * np.cos(
        (2.0 * ys + 1.0) * float(v) * np.pi / 16.0
    )
    return np.sum(arr * basis, axis=(1, 2))


def _build_formula_pool(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    train = np.asarray(X_train, dtype=float).reshape(-1, 8, 8)
    test = np.asarray(X_test, dtype=float).reshape(-1, 8, 8)
    features: list[tuple[np.ndarray, np.ndarray]] = []
    names: list[str] = []
    expressions: list[str] = []
    families: list[str] = []

    def add(train_col: np.ndarray, test_col: np.ndarray, *, name: str, expression: str, family: str) -> None:
        _append_feature(features, names, expressions, families, train_col, test_col, name=name, expression=expression, family=family)

    total_train = np.sum(train, axis=(1, 2))
    total_test = np.sum(test, axis=(1, 2))
    denom_train = np.maximum(total_train, 1.0e-8)
    denom_test = np.maximum(total_test, 1.0e-8)
    rows = np.arange(8, dtype=float).reshape(1, 8, 1)
    cols = np.arange(8, dtype=float).reshape(1, 1, 8)
    row_center_train = np.sum(train * rows, axis=(1, 2)) / denom_train
    row_center_test = np.sum(test * rows, axis=(1, 2)) / denom_test
    col_center_train = np.sum(train * cols, axis=(1, 2)) / denom_train
    col_center_test = np.sum(test * cols, axis=(1, 2)) / denom_test
    add(total_train, total_test, name="total_ink", expression="sum(px[i,j])", family="mass")
    add(row_center_train, row_center_test, name="row_center", expression="sum(i*px[i,j])/sum(px)", family="moment")
    add(col_center_train, col_center_test, name="col_center", expression="sum(j*px[i,j])/sum(px)", family="moment")
    add(
        np.sum(train * ((rows - row_center_train.reshape(-1, 1, 1)) ** 2), axis=(1, 2)) / denom_train,
        np.sum(test * ((rows - row_center_test.reshape(-1, 1, 1)) ** 2), axis=(1, 2)) / denom_test,
        name="row_variance",
        expression="sum((i-row_center)^2*px[i,j])/sum(px)",
        family="moment",
    )
    add(
        np.sum(train * ((cols - col_center_train.reshape(-1, 1, 1)) ** 2), axis=(1, 2)) / denom_train,
        np.sum(test * ((cols - col_center_test.reshape(-1, 1, 1)) ** 2), axis=(1, 2)) / denom_test,
        name="col_variance",
        expression="sum((j-col_center)^2*px[i,j])/sum(px)",
        family="moment",
    )

    row_sum_train = np.sum(train, axis=2)
    row_sum_test = np.sum(test, axis=2)
    col_sum_train = np.sum(train, axis=1)
    col_sum_test = np.sum(test, axis=1)
    for i in range(8):
        add(row_sum_train[:, i], row_sum_test[:, i], name=f"row_{i}_ink", expression=f"sum(px[{i},j])", family="row_projection")
        add(col_sum_train[:, i], col_sum_test[:, i], name=f"col_{i}_ink", expression=f"sum(px[i,{i}])", family="col_projection")

    def patch_feature_name(*, size: int, r: int, c: int, op: str, legacy_op: str = "") -> str:
        if r % size == 0 and c % size == 0 and legacy_op:
            return f"patch{size}x{size}_r{r // size}_c{c // size}_{legacy_op}"
        return f"patch{size}x{size}_rpx{r}_cpx{c}_{op}"

    for size in (2, 4):
        for r in range(0, 8 - size + 1):
            for c in range(0, 8 - size + 1):
                tr_patch = train[:, r : r + size, c : c + size]
                te_patch = test[:, r : r + size, c : c + size]
                patch_label = f"patch{size}x{size}[r={r},c={c}]"
                pool_ops = {
                    "sum": (np.sum(tr_patch, axis=(1, 2)), np.sum(te_patch, axis=(1, 2)), "ink"),
                    "mean": (np.mean(tr_patch, axis=(1, 2)), np.mean(te_patch, axis=(1, 2)), ""),
                    "max": (np.max(tr_patch, axis=(1, 2)), np.max(te_patch, axis=(1, 2)), ""),
                }
                for op, (tr_col, te_col, legacy_op) in pool_ops.items():
                    add(
                        tr_col,
                        te_col,
                        name=patch_feature_name(size=size, r=r, c=c, op=op, legacy_op=legacy_op),
                        expression=f"{op}(px[i,j] for (i,j) in {patch_label})",
                        family="patch_pool",
                    )
                texture_ops = {
                    "var": (np.var(tr_patch, axis=(1, 2)), np.var(te_patch, axis=(1, 2)), "var"),
                    "std": (np.std(tr_patch, axis=(1, 2)), np.std(te_patch, axis=(1, 2)), ""),
                    "range": (
                        np.max(tr_patch, axis=(1, 2)) - np.min(tr_patch, axis=(1, 2)),
                        np.max(te_patch, axis=(1, 2)) - np.min(te_patch, axis=(1, 2)),
                        "",
                    ),
                }
                for op, (tr_col, te_col, legacy_op) in texture_ops.items():
                    add(
                        tr_col,
                        te_col,
                        name=patch_feature_name(size=size, r=r, c=c, op=op, legacy_op=legacy_op),
                        expression=f"{op}(px[i,j] for (i,j) in {patch_label})",
                        family="patch_texture",
                    )

    diff_h_train = np.diff(train, axis=2)
    diff_h_test = np.diff(test, axis=2)
    diff_v_train = np.diff(train, axis=1)
    diff_v_test = np.diff(test, axis=1)
    grad_h_train = np.sum(np.abs(diff_h_train), axis=2)
    grad_h_test = np.sum(np.abs(diff_h_test), axis=2)
    grad_v_train = np.sum(np.abs(diff_v_train), axis=1)
    grad_v_test = np.sum(np.abs(diff_v_test), axis=1)
    signed_h_train = np.sum(diff_h_train, axis=2)
    signed_h_test = np.sum(diff_h_test, axis=2)
    signed_v_train = np.sum(diff_v_train, axis=1)
    signed_v_test = np.sum(diff_v_test, axis=1)
    squared_h_train = np.sum(diff_h_train * diff_h_train, axis=2)
    squared_h_test = np.sum(diff_h_test * diff_h_test, axis=2)
    squared_v_train = np.sum(diff_v_train * diff_v_train, axis=1)
    squared_v_test = np.sum(diff_v_test * diff_v_test, axis=1)
    for i in range(8):
        add(
            grad_h_train[:, i],
            grad_h_test[:, i],
            name=f"row_{i}_horizontal_edge",
            expression=f"sum(abs(px[{i},j+1]-px[{i},j]))",
            family="edge",
        )
        add(
            signed_h_train[:, i],
            signed_h_test[:, i],
            name=f"row_{i}_horizontal_edge_signed",
            expression=f"sum(px[{i},j+1]-px[{i},j])",
            family="edge",
        )
        add(
            squared_h_train[:, i],
            squared_h_test[:, i],
            name=f"row_{i}_horizontal_edge_squared",
            expression=f"sum((px[{i},j+1]-px[{i},j])^2)",
            family="edge",
        )
        add(
            grad_v_train[:, i],
            grad_v_test[:, i],
            name=f"col_{i}_vertical_edge",
            expression=f"sum(abs(px[i+1,{i}]-px[i,{i}]))",
            family="edge",
        )
        add(
            signed_v_train[:, i],
            signed_v_test[:, i],
            name=f"col_{i}_vertical_edge_signed",
            expression=f"sum(px[i+1,{i}]-px[i,{i}])",
            family="edge",
        )
        add(
            squared_v_train[:, i],
            squared_v_test[:, i],
            name=f"col_{i}_vertical_edge_squared",
            expression=f"sum((px[i+1,{i}]-px[i,{i}])^2)",
            family="edge",
        )
    add(np.sum(grad_h_train, axis=1), np.sum(grad_h_test, axis=1), name="global_horizontal_edge", expression="sum(abs(diff_x(px)))", family="edge")
    add(np.sum(signed_h_train, axis=1), np.sum(signed_h_test, axis=1), name="global_horizontal_edge_signed", expression="sum(diff_x(px))", family="edge")
    add(np.sum(squared_h_train, axis=1), np.sum(squared_h_test, axis=1), name="global_horizontal_edge_squared", expression="sum(diff_x(px)^2)", family="edge")
    add(np.sum(grad_v_train, axis=1), np.sum(grad_v_test, axis=1), name="global_vertical_edge", expression="sum(abs(diff_y(px)))", family="edge")
    add(np.sum(signed_v_train, axis=1), np.sum(signed_v_test, axis=1), name="global_vertical_edge_signed", expression="sum(diff_y(px))", family="edge")
    add(np.sum(squared_v_train, axis=1), np.sum(squared_v_test, axis=1), name="global_vertical_edge_squared", expression="sum(diff_y(px)^2)", family="edge")
    add(
        np.sum(np.abs(train[:, :, :4] - train[:, :, 4:][:, :, ::-1]), axis=(1, 2)),
        np.sum(np.abs(test[:, :, :4] - test[:, :, 4:][:, :, ::-1]), axis=(1, 2)),
        name="left_right_symmetry_error",
        expression="sum(abs(left_half(px)-mirror(right_half(px))))",
        family="symmetry",
    )
    add(
        np.sum(np.abs(train[:, :4, :] - train[:, 4:, :][:, ::-1, :]), axis=(1, 2)),
        np.sum(np.abs(test[:, :4, :] - test[:, 4:, :][:, ::-1, :]), axis=(1, 2)),
        name="top_bottom_symmetry_error",
        expression="sum(abs(top_half(px)-mirror(bottom_half(px))))",
        family="symmetry",
    )
    add(np.sum(train[:, 2:6, 2:6], axis=(1, 2)), np.sum(test[:, 2:6, 2:6], axis=(1, 2)), name="center_4x4_ink", expression="sum(px[2:6,2:6])", family="region")
    add(
        np.sum(train, axis=(1, 2)) - np.sum(train[:, 2:6, 2:6], axis=(1, 2)),
        np.sum(test, axis=(1, 2)) - np.sum(test[:, 2:6, 2:6], axis=(1, 2)),
        name="outer_ring_ink",
        expression="sum(px)-sum(px[2:6,2:6])",
        family="region",
    )
    for u in range(4):
        for v in range(4):
            if u == 0 and v == 0:
                continue
            add(
                _dct_basis(train, u, v),
                _dct_basis(test, u, v),
                name=f"dct_u{u}_v{v}",
                expression=f"sum(px[i,j]*cos_basis_u{u}_v{v}[i,j])",
                family="orthogonal_frequency",
            )

    train_cols = [item[0] for item in features]
    test_cols = [item[1] for item in features]
    return (
        np.hstack(train_cols) if train_cols else np.empty((train.shape[0], 0), dtype=float),
        np.hstack(test_cols) if test_cols else np.empty((test.shape[0], 0), dtype=float),
        tuple(names),
        tuple(expressions),
        tuple(families),
    )


def search_image_representation_formulas(
    dataset: ImageClassificationDataset,
    cfg: ImageClassificationConfig,
) -> RepresentationFormulaSearchResult:
    pool_train, pool_test, names, expressions, families = _build_formula_pool(dataset.X_train, dataset.X_test)
    target = _one_vs_rest_target(dataset.y_train)
    scores = np.asarray([_target_corr_score(pool_train[:, j], target) for j in range(pool_train.shape[1])], dtype=float)
    keep_top = min(int(cfg.representation_candidate_keep_top), int(pool_train.shape[1]))
    order = tuple(int(i) for i in np.argsort(-scores)[:keep_top])
    selected: list[int] = []
    max_features = min(int(cfg.representation_max_features), keep_top)
    max_abs_corr = float(cfg.representation_max_pair_abs_corr)
    for idx in order:
        if len(selected) >= max_features:
            break
        if not selected:
            selected.append(idx)
            continue
        corr_max = max(abs(_safe_corr(pool_train[:, idx], pool_train[:, prev])) for prev in selected)
        if corr_max <= max_abs_corr:
            selected.append(idx)
    selected_tuple = tuple(selected)
    selected_set = set(selected_tuple)
    formula_rows: list[dict[str, Any]] = []
    selected_rank = {idx: rank for rank, idx in enumerate(selected_tuple)}
    for rank, idx in enumerate(order):
        formula_rows.append(
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
        pool_train=pool_train,
        pool_test=pool_test,
        selected_train=pool_train[:, selected_tuple] if selected_tuple else np.empty((pool_train.shape[0], 0), dtype=float),
        selected_test=pool_test[:, selected_tuple] if selected_tuple else np.empty((pool_test.shape[0], 0), dtype=float),
        pool_feature_names=names,
        selected_feature_names=tuple(names[i] for i in selected_tuple),
        formula_rows=tuple(formula_rows),
        report={
            "component": "representation_formula_search",
            "search_surface": "mlblack_proxy_ready_candidate_pool",
            "outer_solver_ready": True,
            "candidate_count": int(pool_train.shape[1]),
            "screened_count": int(keep_top),
            "selected_count": int(len(selected_tuple)),
            "selection_policy": "train_target_corr_with_pairwise_redundancy_cap",
            "max_pair_abs_corr": float(max_abs_corr),
            "families": tuple(sorted(set(families))),
        },
    )


__all__ = ["RepresentationFormulaSearchResult", "search_image_representation_formulas"]

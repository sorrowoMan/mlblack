from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ArrayFn = Callable[[np.ndarray], np.ndarray]


def _safe_name(value: Any) -> str:
    return str(value or "").strip()


def _as_1d(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr.reshape(-1)


def _safe_std(value: np.ndarray) -> float:
    std = float(np.nanstd(np.asarray(value, dtype=float)))
    return std if np.isfinite(std) and std > 1e-12 else 1.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    left = _as_1d(a)
    right = _as_1d(b)
    if left.size != right.size or left.size < 3:
        return 0.0
    if float(np.nanstd(left)) <= 1e-12 or float(np.nanstd(right)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def _safe_corr_target(values: np.ndarray, target: np.ndarray) -> float:
    arr = _as_1d(values)
    tgt = np.asarray(target, dtype=float)
    if tgt.ndim == 1:
        return _safe_corr(arr, tgt)
    if tgt.ndim != 2 or tgt.shape[0] != arr.shape[0]:
        return 0.0
    scores = [_safe_corr(arr, tgt[:, j]) for j in range(tgt.shape[1])]
    if not scores:
        return 0.0
    best = max(scores, key=lambda value: abs(float(value)))
    return float(best)


def _stable_scale(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    train = np.asarray(train_values, dtype=float)
    arr = np.asarray(values, dtype=float)
    mean = float(np.nanmean(train))
    std = _safe_std(train)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class OrthogonalSourceConfig:
    max_sources: int = 10
    candidate_keep_top: int = 160
    max_pair_abs_corr: float = 0.72
    min_abs_target_corr: float = 0.02
    residual_corr_weight: float = 0.75
    target_corr_weight: float = 0.45
    stability_weight: float = 0.20
    include_raw: bool = True
    include_unary: bool = True
    include_pairwise: bool = True
    include_triple_ratio: bool = True
    include_hinge: bool = True
    include_exp_ratio: bool = True
    safe_eps: float = 1e-3
    hinge_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    standardize: bool = True
    target_task: str = "auto"  # auto | regression | classification
    max_auto_classes: int = 20


@dataclass(frozen=True)
class _Candidate:
    name: str
    expression: str
    family: str
    support_features: tuple[str, ...]
    fn: ArrayFn


@dataclass(frozen=True)
class OrthogonalSourceResult:
    train_basis: np.ndarray
    test_basis: np.ndarray
    source_rows: tuple[dict[str, Any], ...]
    report: dict[str, Any] = field(default_factory=dict)


class OrthogonalSourceLayer:
    """Model-agnostic source governance layer for downstream learners.

    The layer generates semantically tagged candidate source objects, then greedily
    selects target-relevant sources under pairwise correlation and residual-gain
    pressure. It deliberately stops at representation construction: downstream
    symbolic, linear, tree, or boosting learners consume the returned basis matrix.
    """

    def __init__(self, config: OrthogonalSourceConfig | None = None) -> None:
        self.config = config or OrthogonalSourceConfig()

    def fit_transform(
        self,
        *,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> OrthogonalSourceResult:
        X_tr = np.asarray(X_train, dtype=float)
        X_te = np.asarray(X_test, dtype=float)
        target_matrix, target_task = self._target_matrix(y_train, dict(metadata or {}))
        names = tuple(_safe_name(name) or f"x{i}" for i, name in enumerate(tuple(feature_names)))
        candidates = self._build_candidates(X_tr, names, dict(metadata or {}))

        scored = []
        for candidate in candidates:
            try:
                raw_train = np.asarray(candidate.fn(X_tr), dtype=float).reshape(-1)
                raw_test = np.asarray(candidate.fn(X_te), dtype=float).reshape(-1)
            except Exception:
                continue
            if raw_train.shape[0] != X_tr.shape[0] or raw_test.shape[0] != X_te.shape[0]:
                continue
            if not np.all(np.isfinite(np.nan_to_num(raw_train, nan=0.0, posinf=0.0, neginf=0.0))):
                continue
            train_col = _stable_scale(raw_train, raw_train) if self.config.standardize else raw_train
            test_col = _stable_scale(raw_train, raw_test) if self.config.standardize else raw_test
            target_corr = _safe_corr_target(train_col, target_matrix)
            stability = self._split_stability(train_col, target_matrix)
            if abs(target_corr) < float(self.config.min_abs_target_corr):
                continue
            scored.append(
                {
                    "candidate": candidate,
                    "train": train_col,
                    "test": test_col,
                    "target_corr": float(target_corr),
                    "stability": float(stability),
                    "base_score": float(abs(target_corr) * (0.8 + 0.2 * max(0.0, stability))),
                }
            )

        scored.sort(key=lambda row: float(row["base_score"]), reverse=True)
        pool = scored[: max(1, int(self.config.candidate_keep_top))]
        selected: list[dict[str, Any]] = []
        residual = target_matrix - np.mean(target_matrix, axis=0, keepdims=True)

        while pool and len(selected) < max(1, int(self.config.max_sources)):
            best_idx = -1
            best_score = -float("inf")
            for idx, row in enumerate(pool):
                train_col = np.asarray(row["train"], dtype=float)
                if self._violates_pair_corr(train_col, selected):
                    continue
                residual_corr = _safe_corr_target(train_col, residual)
                score = (
                    float(self.config.target_corr_weight) * abs(float(row["target_corr"]))
                    + float(self.config.residual_corr_weight) * abs(float(residual_corr))
                    + float(self.config.stability_weight) * float(row["stability"])
                )
                if score > best_score:
                    best_score = float(score)
                    best_idx = int(idx)
                    row["residual_corr"] = float(residual_corr)
                    row["selection_score"] = float(score)
            if best_idx < 0:
                break
            picked = pool.pop(best_idx)
            selected.append(picked)
            matrix = np.column_stack([np.asarray(item["train"], dtype=float) for item in selected])
            coef, *_ = np.linalg.lstsq(matrix, target_matrix, rcond=None)
            residual = target_matrix - matrix @ coef

        if not selected and pool:
            selected.append(pool[0])

        train_basis = (
            np.column_stack([np.asarray(item["train"], dtype=float) for item in selected])
            if selected
            else np.zeros((X_tr.shape[0], 0), dtype=float)
        )
        test_basis = (
            np.column_stack([np.asarray(item["test"], dtype=float) for item in selected])
            if selected
            else np.zeros((X_te.shape[0], 0), dtype=float)
        )
        rows = tuple(self._row_from_selection(idx, item, selected) for idx, item in enumerate(selected))
        report = self._build_report(
            train_basis,
            rows,
            candidate_count=len(candidates),
            screened_count=len(pool) + len(selected),
            target_task=target_task,
            target_dim=target_matrix.shape[1],
        )
        return OrthogonalSourceResult(train_basis=train_basis, test_basis=test_basis, source_rows=rows, report=report)

    def _target_matrix(self, y_train: np.ndarray, metadata: Mapping[str, Any]) -> tuple[np.ndarray, str]:
        y = _as_1d(np.asarray(y_train, dtype=float))
        requested = str(metadata.get("target_task", self.config.target_task) or self.config.target_task).strip().lower()
        if requested not in {"auto", "regression", "classification"}:
            requested = "auto"

        unique = np.unique(y[np.isfinite(y)])
        integer_like = bool(unique.size > 0 and np.allclose(unique, np.round(unique), atol=1e-9))
        auto_classification = bool(
            requested == "auto"
            and integer_like
            and 2 <= unique.size <= max(2, int(self.config.max_auto_classes))
        )
        is_classification = bool(requested == "classification" or auto_classification)
        if not is_classification:
            return y.reshape(-1, 1), "regression"

        classes = tuple(float(v) for v in unique)
        class_to_idx = {float(cls): idx for idx, cls in enumerate(classes)}
        out = np.zeros((y.shape[0], len(classes)), dtype=float)
        for row_idx, value in enumerate(y):
            idx = class_to_idx.get(float(value))
            if idx is not None:
                out[row_idx, idx] = 1.0
        return out, "classification"

    def _build_candidates(
        self,
        X_train: np.ndarray,
        feature_names: tuple[str, ...],
        metadata: Mapping[str, Any],
    ) -> list[_Candidate]:
        cfg = self.config
        eps = float(cfg.safe_eps)
        candidates: list[_Candidate] = []
        n_features = len(feature_names)

        if cfg.include_raw:
            for i, name in enumerate(feature_names):
                candidates.append(
                    _Candidate(
                        name=name,
                        expression=name,
                        family="raw",
                        support_features=(name,),
                        fn=lambda X, i=i: X[:, i],
                    )
                )

        if cfg.include_unary:
            for i, name in enumerate(feature_names):
                candidates.extend(
                    [
                        _Candidate(f"sin_{name}", f"sin({name})", "periodic", (name,), lambda X, i=i: np.sin(X[:, i])),
                        _Candidate(f"cos_{name}", f"cos({name})", "periodic", (name,), lambda X, i=i: np.cos(X[:, i])),
                        _Candidate(f"square_{name}", f"({name})^2", "power", (name,), lambda X, i=i: X[:, i] ** 2),
                        _Candidate(
                            f"reciprocal_{name}",
                            f"1/safe_abs({name})",
                            "reciprocal",
                            (name,),
                            lambda X, i=i, eps=eps: 1.0 / (np.abs(X[:, i]) + eps),
                        ),
                    ]
                )

        if cfg.include_hinge:
            quantiles = tuple(float(q) for q in tuple(cfg.hinge_quantiles))
            for i, name in enumerate(feature_names):
                cuts = tuple(float(np.quantile(X_train[:, i], q)) for q in quantiles)
                for q, cut in zip(quantiles, cuts):
                    suffix = str(q).replace(".", "p")
                    candidates.append(
                        _Candidate(
                            f"hinge_pos_{name}_{suffix}",
                            f"max(0,{name}-{cut:.6g})",
                            "piecewise_gate",
                            (name,),
                            lambda X, i=i, cut=cut: np.maximum(X[:, i] - cut, 0.0),
                        )
                    )
                    candidates.append(
                        _Candidate(
                            f"hinge_neg_{name}_{suffix}",
                            f"max(0,{cut:.6g}-{name})",
                            "piecewise_gate",
                            (name,),
                            lambda X, i=i, cut=cut: np.maximum(cut - X[:, i], 0.0),
                        )
                    )

        if cfg.include_pairwise:
            for i, j in permutations(range(n_features), 2):
                left, right = feature_names[i], feature_names[j]
                candidates.append(
                    _Candidate(
                        f"ratio_{left}_over_{right}",
                        f"{left}/safe_abs({right})",
                        "ratio",
                        (left, right),
                        lambda X, i=i, j=j, eps=eps: X[:, i] / (np.abs(X[:, j]) + eps),
                    )
                )
            for i, j in combinations(range(n_features), 2):
                left, right = feature_names[i], feature_names[j]
                candidates.append(
                    _Candidate(
                        f"product_{left}_{right}",
                        f"{left}*{right}",
                        "product",
                        (left, right),
                        lambda X, i=i, j=j: X[:, i] * X[:, j],
                    )
                )

        if cfg.include_triple_ratio and n_features <= 10:
            for i, j, k in permutations(range(n_features), 3):
                if i >= j:
                    continue
                a, b, c = feature_names[i], feature_names[j], feature_names[k]
                candidates.append(
                    _Candidate(
                        f"product_ratio_{a}_{b}_over_{c}",
                        f"({a}*{b})/safe_abs({c})",
                        "product_ratio",
                        (a, b, c),
                        lambda X, i=i, j=j, k=k, eps=eps: (X[:, i] * X[:, j]) / (np.abs(X[:, k]) + eps),
                    )
                )

        if cfg.include_exp_ratio:
            for i, j in permutations(range(n_features), 2):
                left, right = feature_names[i], feature_names[j]
                candidates.append(
                    _Candidate(
                        f"exp_neg_ratio_{left}_over_{right}",
                        f"exp(-{left}/safe_abs({right}))",
                        "exp_ratio",
                        (left, right),
                        lambda X, i=i, j=j, eps=eps: np.exp(
                            np.clip(-X[:, i] / (np.abs(X[:, j]) + eps), -40.0, 40.0)
                        ),
                    )
                )

        # Metadata-guided mechanistic groups receive explicit canonical candidates.
        hints = dict(metadata.get("orchestrator_hints", {}) or {})
        overrides = dict(hints.get("trainer_params_overrides", {}) or {})
        for group in tuple(overrides.get("orth_mechanistic_feature_groups", ()) or ()):
            names = tuple(str(item) for item in tuple(group) if str(item) in feature_names)
            if len(names) == 2:
                i, j = feature_names.index(names[0]), feature_names.index(names[1])
                candidates.append(
                    _Candidate(
                        f"metadata_exp_neg_ratio_{names[0]}_over_{names[1]}",
                        f"exp(-{names[0]}/safe_abs({names[1]}))",
                        "metadata_exp_ratio",
                        names,
                        lambda X, i=i, j=j, eps=eps: np.exp(
                            np.clip(-X[:, i] / (np.abs(X[:, j]) + eps), -40.0, 40.0)
                        ),
                    )
                )
            if len(names) == 3:
                i, j, k = (feature_names.index(names[0]), feature_names.index(names[1]), feature_names.index(names[2]))
                candidates.append(
                    _Candidate(
                        f"metadata_product_ratio_{names[0]}_{names[1]}_over_{names[2]}",
                        f"({names[0]}*{names[1]})/safe_abs({names[2]})",
                        "metadata_product_ratio",
                        names,
                        lambda X, i=i, j=j, k=k, eps=eps: (X[:, i] * X[:, j]) / (np.abs(X[:, k]) + eps),
                    )
                )
        return candidates

    def _split_stability(self, values: np.ndarray, y: np.ndarray) -> float:
        arr = _as_1d(values)
        target = np.asarray(y, dtype=float)
        if arr.size < 8:
            return 0.0
        mid = arr.size // 2
        c1 = _safe_corr_target(arr[:mid], target[:mid])
        c2 = _safe_corr_target(arr[mid:], target[mid:])
        sign_ok = 1.0 if c1 == 0.0 or c2 == 0.0 or np.sign(c1) == np.sign(c2) else 0.0
        mag = 1.0 - min(1.0, abs(abs(c1) - abs(c2)))
        return float(0.65 * sign_ok + 0.35 * max(0.0, mag))

    def _violates_pair_corr(self, values: np.ndarray, selected: Sequence[Mapping[str, Any]]) -> bool:
        threshold = float(self.config.max_pair_abs_corr)
        for item in selected:
            if abs(_safe_corr(values, np.asarray(item["train"], dtype=float))) > threshold:
                return True
        return False

    def _row_from_selection(
        self,
        idx: int,
        item: Mapping[str, Any],
        selected: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        candidate = item["candidate"]
        pair_corrs = [
            abs(_safe_corr(np.asarray(item["train"], dtype=float), np.asarray(other["train"], dtype=float)))
            for other in selected
            if other is not item
        ]
        return {
            "rank": int(idx),
            "name": str(candidate.name),
            "expression": str(candidate.expression),
            "family": str(candidate.family),
            "support_features": tuple(candidate.support_features),
            "target_corr": float(item.get("target_corr", 0.0)),
            "residual_corr": float(item.get("residual_corr", 0.0)),
            "stability": float(item.get("stability", 0.0)),
            "selection_score": float(item.get("selection_score", item.get("base_score", 0.0))),
            "max_pair_abs_corr": 0.0 if not pair_corrs else float(max(pair_corrs)),
        }

    def _build_report(
        self,
        train_basis: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        *,
        candidate_count: int,
        screened_count: int,
        target_task: str,
        target_dim: int,
    ) -> dict[str, Any]:
        if train_basis.shape[1] >= 2:
            corr = np.corrcoef(train_basis, rowvar=False)
            upper = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
            pair_abs_corr_mean = float(np.nanmean(upper)) if upper.size else 0.0
            pair_abs_corr_max = float(np.nanmax(upper)) if upper.size else 0.0
        else:
            pair_abs_corr_mean = 0.0
            pair_abs_corr_max = 0.0
        return {
            "component": "orthogonal_source_layer",
            "target_task": str(target_task),
            "target_dim": int(target_dim),
            "candidate_count": int(candidate_count),
            "screened_count": int(screened_count),
            "selected_source_count": int(len(tuple(rows))),
            "pair_abs_corr_mean": pair_abs_corr_mean,
            "pair_abs_corr_max": pair_abs_corr_max,
            "mean_source_stability": float(np.mean([float(row.get("stability", 0.0)) for row in rows])) if rows else 0.0,
            "families": tuple(sorted({str(row.get("family", "")) for row in rows if str(row.get("family", "")).strip()})),
        }

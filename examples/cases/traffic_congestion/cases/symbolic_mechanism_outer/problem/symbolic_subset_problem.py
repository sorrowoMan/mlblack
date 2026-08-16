from __future__ import annotations

from ..runtime.legacy_imports import *
from ..artifacts.serialization import *
from ..evaluation.metrics import *
from ..pipeline.inner_fit import *
from ..pipeline.splits import *
from ..pipeline.regimes import *
from ..pipeline.candidate_pool import *

class SymbolicSubsetSelectionProblem(BlackBoxProblem):
    def __init__(
        self,
        *,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        candidates: Sequence[CandidateTerm],
        max_terms: int,
        ridge_l2: float,
        rolling_folds: int,
        rolling_val_ratio: float,
        min_train: int,
        strict4_branch_mode: bool,
        strict4_gate_idx: tuple[int, int, int, int] | None,
        strict4_min_branch_train: int,
        strict4_branch_parallel_workers: int,
        inner_opt_enabled: bool,
        inner_opt_adam_steps: int,
        inner_opt_adam_lr: float,
        inner_opt_lbfgs_steps: int,
        inner_opt_lbfgs_lr: float,
        inner_opt_accept_rmse_tol: float,
        inner_opt_accept_rel_tol: float,
        inner_opt_guard_patience: int,
        inner_opt_guard_check_interval: int,
        inner_opt_alt_freeze_readout: bool,
        inner_opt_grad_clip_norm: float,
        inner_opt_residual_clip_q: float,
        graph_cache: ExpressionGraphCache | None = None,
    ) -> None:
        self.X_fit = np.asarray(X_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        self.candidates = list(candidates)
        self.families = tuple(sorted({str(c.family) for c in self.candidates}))
        self.family_to_idx = {str(v): int(i) for i, v in enumerate(self.families)}
        self.max_terms = int(max(2, max_terms))
        self.base_ridge_l2 = float(max(0.0, ridge_l2))
        self.strict4_branch_mode = bool(strict4_branch_mode)
        self.strict4_gate_idx = strict4_gate_idx
        self.base_strict4_min_branch_train = int(max(8, strict4_min_branch_train))
        self.strict4_branch_parallel_workers = int(max(1, strict4_branch_parallel_workers))
        self.inner_opt_enabled = bool(inner_opt_enabled)
        self.inner_opt_adam_steps = int(max(0, inner_opt_adam_steps))
        self.inner_opt_adam_lr = float(max(1e-8, inner_opt_adam_lr))
        self.inner_opt_lbfgs_steps = int(max(0, inner_opt_lbfgs_steps))
        self.inner_opt_lbfgs_lr = float(max(1e-8, inner_opt_lbfgs_lr))
        self.inner_opt_accept_rmse_tol = float(max(0.0, inner_opt_accept_rmse_tol))
        self.inner_opt_accept_rel_tol = float(max(0.0, inner_opt_accept_rel_tol))
        self.inner_opt_guard_patience = int(max(1, inner_opt_guard_patience))
        self.inner_opt_guard_check_interval = int(max(1, inner_opt_guard_check_interval))
        self.inner_opt_alt_freeze_readout = bool(inner_opt_alt_freeze_readout)
        self.inner_opt_grad_clip_norm = float(max(0.0, inner_opt_grad_clip_norm))
        self.inner_opt_residual_clip_q = float(np.clip(inner_opt_residual_clip_q, 0.70, 0.999))
        self.graph_cache = graph_cache
        self.splits = _rolling_splits(
            int(self.X_fit.shape[0]),
            folds=int(max(1, rolling_folds)),
            val_ratio=float(rolling_val_ratio),
            min_train=int(max(128, min_train)),
        )
        self._cache: dict[tuple[tuple[int, ...], str], tuple[np.ndarray, dict[str, Any]]] = {}

        n_terms = int(len(self.candidates))
        n_fam = int(len(self.families))
        self.n_hyper_genes = 11
        bounds = (
            [(-1.0, 1.0) for _ in range(n_terms)]
            + [(-0.8, 0.8) for _ in range(n_fam)]
            + [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
            + [(0.0, 1.0) for _ in range(self.n_hyper_genes)]
        )
        super().__init__(
            name="SymbolicSubsetSelectionProblem",
            dimension=int(n_terms + n_fam + 3 + self.n_hyper_genes),
            bounds=bounds,
            objectives=["minimize", "minimize", "minimize"],
        )

    def _decode(self, x: np.ndarray) -> tuple[list[int], int, dict[str, Any]]:
        z = np.asarray(x, dtype=float).reshape(-1)
        n_terms = int(len(self.candidates))
        n_fam = int(len(self.families))
        raw_scores = np.asarray(z[:n_terms], dtype=float)
        family_bias = np.asarray(z[n_terms : n_terms + n_fam], dtype=float)
        k_gene = float(np.clip(z[n_terms + n_fam], 0.0, 1.0))
        thresh_gene = float(np.clip(z[n_terms + n_fam + 1], 0.0, 1.0))
        inter_gene = float(np.clip(z[n_terms + n_fam + 2], 0.0, 1.0))
        hyper = np.asarray(z[n_terms + n_fam + 3 :], dtype=float)
        if hyper.size < self.n_hyper_genes:
            hyper = np.pad(hyper, (0, int(self.n_hyper_genes - hyper.size)), constant_values=0.5)

        prior_corr_w = float(0.05 + 0.85 * float(np.clip(hyper[0], 0.0, 1.0)))
        family_bias_scale = float(0.10 + 1.40 * float(np.clip(hyper[1], 0.0, 1.0)))
        tuned_l2 = float(10.0 ** (-8.0 + 6.0 * float(np.clip(hyper[2], 0.0, 1.0))))
        complexity_scale = float(0.30 + 1.50 * float(np.clip(hyper[3], 0.0, 1.0)))
        family_penalty_scale = float(0.20 + 1.20 * float(np.clip(hyper[4], 0.0, 1.0)))
        feature_penalty_scale = float(0.20 + 1.20 * float(np.clip(hyper[5], 0.0, 1.0)))
        drift_weight = float(0.05 + 0.40 * float(np.clip(hyper[6], 0.0, 1.0)))
        strict4_min_train_ratio = float(0.02 + 0.18 * float(np.clip(hyper[7], 0.0, 1.0)))
        q_low = float(0.15 + 0.45 * float(np.clip(hyper[8], 0.0, 1.0)))
        q_span = float(0.20 + 0.70 * float(np.clip(hyper[9], 0.0, 1.0)))
        inter_floor_ratio = float(0.50 * float(np.clip(hyper[10], 0.0, 1.0)))

        k = int(round(2 + k_gene * (self.max_terms - 2)))
        k = int(np.clip(k, 2, self.max_terms))

        adj = np.asarray(raw_scores, dtype=float).copy()
        for i, cand in enumerate(self.candidates):
            fam_idx = int(self.family_to_idx.get(str(cand.family), 0))
            adj[i] = float(
                adj[i]
                + prior_corr_w * float(cand.prior_corr)
                + family_bias_scale * float(family_bias[fam_idx])
            )

        q = float(np.clip(q_low + q_span * thresh_gene, 0.05, 0.98))
        cut = float(np.quantile(adj, q))
        active = [int(i) for i in range(adj.size) if float(adj[i]) >= cut]
        if len(active) < 2:
            active = list(range(int(adj.size)))
        order = sorted(active, key=lambda i: float(adj[i]), reverse=True)

        inter_cap = int(max(1, round((inter_floor_ratio + (1.0 - inter_floor_ratio) * inter_gene) * k)))
        inter_count = 0
        picked: list[int] = []
        for i in order:
            fam = str(self.candidates[i].family)
            if fam == "interaction" and inter_count >= inter_cap:
                continue
            picked.append(int(i))
            if fam == "interaction":
                inter_count += 1
            if len(picked) >= k:
                break

        if len(picked) < 2:
            order_all = list(np.argsort(-adj))
            picked = [int(i) for i in order_all[: max(2, k)]]

        # Encourage at least one linear anchor when possible.
        if not any(str(self.candidates[i].family) == "linear" for i in picked):
            linear_idx = [i for i in range(len(self.candidates)) if str(self.candidates[i].family) == "linear"]
            if linear_idx:
                best_linear = int(max(linear_idx, key=lambda i: float(adj[i])))
                picked = [best_linear] + [i for i in picked if i != best_linear]
                picked = picked[:k]

        meta = {
            "k": int(k),
            "threshold_q": float(q),
            "threshold_cut": float(cut),
            "interaction_cap": int(inter_cap),
            "interaction_count": int(sum(1 for i in picked if str(self.candidates[i].family) == "interaction")),
            "tuned_l2": float(tuned_l2),
            "complexity_scale": float(complexity_scale),
            "family_penalty_scale": float(family_penalty_scale),
            "feature_penalty_scale": float(feature_penalty_scale),
            "drift_weight": float(drift_weight),
            "strict4_min_train_ratio": float(strict4_min_train_ratio),
            "prior_corr_w": float(prior_corr_w),
            "family_bias_scale": float(family_bias_scale),
        }
        if len(picked) < 2:
            picked = [int(i) for i in list(np.argsort(-adj))[:2]]
        return picked, k, meta

    @staticmethod
    def _cache_sig(meta: Mapping[str, Any]) -> str:
        keys = (
            "tuned_l2",
            "complexity_scale",
            "family_penalty_scale",
            "feature_penalty_scale",
            "drift_weight",
            "strict4_min_train_ratio",
            "prior_corr_w",
            "family_bias_scale",
            "threshold_q",
            "interaction_cap",
            "k",
        )
        out: dict[str, Any] = {}
        for k in keys:
            v = meta.get(k)
            if isinstance(v, float):
                out[str(k)] = round(float(v), 7)
            else:
                out[str(k)] = v
        return json.dumps(out, sort_keys=True, separators=(",", ":"))

    def _eval_fold_global(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
    ) -> dict[str, Any]:
        fit = _three_layer_fit_predict(
            genome=genome,
            X_train=self.X_fit[tr_idx],
            y_train=self.y_fit[tr_idx],
            X_eval=self.X_fit[va_idx],
            y_eval=self.y_fit[va_idx],
            l2=float(max(0.0, l2)),
            inner_opt_enabled=bool(self.inner_opt_enabled),
            inner_opt_adam_steps=int(self.inner_opt_adam_steps),
            inner_opt_adam_lr=float(self.inner_opt_adam_lr),
            inner_opt_lbfgs_steps=int(self.inner_opt_lbfgs_steps),
            inner_opt_lbfgs_lr=float(self.inner_opt_lbfgs_lr),
            inner_opt_accept_rmse_tol=float(self.inner_opt_accept_rmse_tol),
            inner_opt_accept_rel_tol=float(self.inner_opt_accept_rel_tol),
            inner_opt_guard_patience=int(self.inner_opt_guard_patience),
            inner_opt_guard_check_interval=int(self.inner_opt_guard_check_interval),
            inner_opt_alt_freeze_readout=bool(self.inner_opt_alt_freeze_readout),
            inner_opt_grad_clip_norm=float(self.inner_opt_grad_clip_norm),
            inner_opt_residual_clip_q=float(self.inner_opt_residual_clip_q),
        )
        m_eval = dict(fit.get("metrics_eval", {}))
        return {
            "rmse": float(m_eval.get("rmse", float("inf"))),
            "mode": "global",
            "branch_detail": {"inner_opt_info": _jsonable(fit.get("inner_opt_info", {}))},
        }

    def _eval_fold_strict4(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
        strict4_min_branch_train: int,
    ) -> dict[str, Any]:
        if not self.strict4_branch_mode or self.strict4_gate_idx is None:
            return self._eval_fold_global(genome, tr_idx, va_idx, l2=l2)

        Xtr = np.asarray(self.X_fit[tr_idx], dtype=float)
        ytr = np.asarray(self.y_fit[tr_idx], dtype=float)
        Xva = np.asarray(self.X_fit[va_idx], dtype=float)
        yva = np.asarray(self.y_fit[va_idx], dtype=float)

        keys_tr = _strict4_keys_from_X(Xtr, self.strict4_gate_idx)
        keys_va = _strict4_keys_from_X(Xva, self.strict4_gate_idx)

        idx_tr_by_key: dict[tuple[int, int, int, int], np.ndarray] = {}
        idx_va_by_key: dict[tuple[int, int, int, int], np.ndarray] = {}
        for k in STRICT4_REGIME_ORDER:
            idx_tr_by_key[k] = np.asarray([i for i, kk in enumerate(keys_tr) if kk == k], dtype=int)
            idx_va_by_key[k] = np.asarray([i for i, kk in enumerate(keys_va) if kk == k], dtype=int)

        fit_global = _three_layer_fit_predict(
            genome=genome,
            X_train=Xtr,
            y_train=ytr,
            X_eval=Xva,
            y_eval=yva,
            l2=float(max(0.0, l2)),
            inner_opt_enabled=bool(self.inner_opt_enabled),
            inner_opt_adam_steps=int(self.inner_opt_adam_steps),
            inner_opt_adam_lr=float(self.inner_opt_adam_lr),
            inner_opt_lbfgs_steps=int(self.inner_opt_lbfgs_steps),
            inner_opt_lbfgs_lr=float(self.inner_opt_lbfgs_lr),
            inner_opt_accept_rmse_tol=float(self.inner_opt_accept_rmse_tol),
            inner_opt_accept_rel_tol=float(self.inner_opt_accept_rel_tol),
            inner_opt_guard_patience=int(self.inner_opt_guard_patience),
            inner_opt_guard_check_interval=int(self.inner_opt_guard_check_interval),
            inner_opt_alt_freeze_readout=bool(self.inner_opt_alt_freeze_readout),
            inner_opt_grad_clip_norm=float(self.inner_opt_grad_clip_norm),
            inner_opt_residual_clip_q=float(self.inner_opt_residual_clip_q),
        )
        pred_global = np.asarray(fit_global.get("pred_eval"), dtype=float).reshape(-1, 1)

        pred_va = np.asarray(pred_global, dtype=float).copy()
        branch_rmse: dict[str, float] = {}
        branch_used_train: dict[str, int] = {}
        branch_used_fallback: dict[str, bool] = {}

        def _fit_branch(k: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], np.ndarray | None, bool]:
            tr_local = np.asarray(idx_tr_by_key[k], dtype=int)
            va_local = np.asarray(idx_va_by_key[k], dtype=int)
            if int(va_local.size) <= 0:
                return k, None, True
            if int(tr_local.size) < int(strict4_min_branch_train):
                return k, None, True
            fit = _three_layer_fit_predict(
                genome=genome,
                X_train=Xtr[tr_local],
                y_train=ytr[tr_local],
                X_eval=Xva[va_local],
                y_eval=yva[va_local],
                l2=float(max(0.0, l2)),
                inner_opt_enabled=bool(self.inner_opt_enabled),
                inner_opt_adam_steps=int(self.inner_opt_adam_steps),
                inner_opt_adam_lr=float(self.inner_opt_adam_lr),
                inner_opt_lbfgs_steps=int(self.inner_opt_lbfgs_steps),
                inner_opt_lbfgs_lr=float(self.inner_opt_lbfgs_lr),
                inner_opt_accept_rmse_tol=float(self.inner_opt_accept_rmse_tol),
                inner_opt_accept_rel_tol=float(self.inner_opt_accept_rel_tol),
                inner_opt_guard_patience=int(self.inner_opt_guard_patience),
                inner_opt_guard_check_interval=int(self.inner_opt_guard_check_interval),
                inner_opt_alt_freeze_readout=bool(self.inner_opt_alt_freeze_readout),
                inner_opt_grad_clip_norm=float(self.inner_opt_grad_clip_norm),
                inner_opt_residual_clip_q=float(self.inner_opt_residual_clip_q),
            )
            pred = np.asarray(fit.get("pred_eval"), dtype=float).reshape(-1, 1)
            return k, pred, False

        n_workers = int(max(1, min(self.strict4_branch_parallel_workers, len(STRICT4_REGIME_ORDER))))
        if n_workers <= 1:
            branch_results = [_fit_branch(k) for k in STRICT4_REGIME_ORDER]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
                fut = [ex.submit(_fit_branch, k) for k in STRICT4_REGIME_ORDER]
                branch_results = [f.result() for f in fut]

        for k, pred_k, fallback in branch_results:
            va_local = np.asarray(idx_va_by_key[k], dtype=int)
            tr_local = np.asarray(idx_tr_by_key[k], dtype=int)
            if int(va_local.size) <= 0:
                continue
            if pred_k is not None and not bool(fallback):
                pred_va[va_local] = np.asarray(pred_k, dtype=float)
            yk = np.asarray(yva[va_local], dtype=float).reshape(-1)
            pk = np.asarray(pred_va[va_local], dtype=float).reshape(-1)
            branch_rmse[str(k)] = float(_rmse(yk, pk))
            branch_used_train[str(k)] = int(tr_local.size)
            branch_used_fallback[str(k)] = bool(fallback)

        rmse = float(_rmse(yva, pred_va))
        return {
            "rmse": float(rmse),
            "mode": "strict4_branch",
            "branch_detail": {
                "branch_rmse": dict(branch_rmse),
                "branch_train_size": dict(branch_used_train),
                "branch_fallback": dict(branch_used_fallback),
            },
        }

    def _evaluate_subset(self, subset_idx: Sequence[int], meta: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        key_subset = tuple(sorted(int(i) for i in subset_idx))
        key = (key_subset, self._cache_sig(meta))
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        genome = [{"name": self.candidates[i].name, "expr": dict(self.candidates[i].expr)} for i in key_subset]
        tuned_l2 = float(max(0.0, meta.get("tuned_l2", self.base_ridge_l2)))
        strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))
        complexity_scale = float(max(0.05, meta.get("complexity_scale", 1.0)))
        family_penalty_scale = float(max(0.05, meta.get("family_penalty_scale", 1.0)))
        feature_penalty_scale = float(max(0.05, meta.get("feature_penalty_scale", 1.0)))
        drift_weight = float(max(0.0, meta.get("drift_weight", 0.15)))

        fold_rmse: list[float] = []
        fold_branch: list[dict[str, Any]] = []
        for fold_id, (tr_idx, va_idx) in enumerate(self.splits):
            tr_arr = np.asarray(tr_idx, dtype=int)
            va_arr = np.asarray(va_idx, dtype=int)
            strict4_min_train = int(
                max(
                    self.base_strict4_min_branch_train,
                    round(float(strict4_ratio) * float(tr_arr.size)),
                )
            )
            fold_res = self._eval_fold_strict4(
                genome,
                tr_idx=tr_arr,
                va_idx=va_arr,
                l2=tuned_l2,
                strict4_min_branch_train=strict4_min_train,
            )
            fold_rmse.append(float(fold_res["rmse"]))
            fold_branch.append(dict(fold_res.get("branch_detail", {})))

        rmse_mean = float(np.mean(fold_rmse))
        rmse_std = float(np.std(fold_rmse))
        rmse_drift = float(np.mean(np.abs(np.diff(np.asarray(fold_rmse, dtype=float))))) if len(fold_rmse) >= 2 else 0.0
        complexity = float(sum(float(self.candidates[i].complexity) for i in key_subset))

        fam_counts: dict[str, int] = {}
        feat_counts: dict[int, int] = {}
        for i in key_subset:
            c = self.candidates[i]
            fam_counts[str(c.family)] = int(fam_counts.get(str(c.family), 0) + 1)
            for f in c.features:
                feat_counts[int(f)] = int(feat_counts.get(int(f), 0) + 1)
        fam_share = np.asarray([float(v) for v in fam_counts.values()], dtype=float)
        if fam_share.size > 0:
            fam_share = fam_share / float(np.sum(fam_share))
        feat_share = np.asarray([float(v) for v in feat_counts.values()], dtype=float)
        if feat_share.size > 0:
            feat_share = feat_share / float(np.sum(feat_share))
        fam_concentration = float(np.sum(fam_share**2)) if fam_share.size > 0 else 1.0
        feat_concentration = float(np.sum(feat_share**2)) if feat_share.size > 0 else 1.0

        obj_accuracy = float(rmse_mean)
        obj_stability = float(rmse_std + drift_weight * rmse_drift)
        obj_complexity = float(
            complexity_scale * (complexity / max(1.0, float(self.max_terms)))
            + family_penalty_scale * fam_concentration
            + feature_penalty_scale * feat_concentration
        )
        out_obj = np.asarray([obj_accuracy, obj_stability, obj_complexity], dtype=float)
        detail = {
            "subset_size": int(len(key_subset)),
            "subset_idx": [int(i) for i in key_subset],
            "subset_names": [self.candidates[i].name for i in key_subset],
            "subset_families": [self.candidates[i].family for i in key_subset],
            "fold_rmse": [float(v) for v in fold_rmse],
            "fold_branch_detail": _jsonable(fold_branch),
            "rmse_mean": float(rmse_mean),
            "rmse_std": float(rmse_std),
            "rmse_drift": float(rmse_drift),
            "complexity_raw": float(complexity),
            "family_concentration": float(fam_concentration),
            "feature_concentration": float(feat_concentration),
            "tuned_l2": float(tuned_l2),
            "strict4_min_train_ratio": float(strict4_ratio),
            "complexity_scale": float(complexity_scale),
            "family_penalty_scale": float(family_penalty_scale),
            "feature_penalty_scale": float(feature_penalty_scale),
            "drift_weight": float(drift_weight),
            "decode_meta": _jsonable(dict(meta)),
        }
        self._cache[key] = (out_obj, detail)
        return out_obj, detail

    @staticmethod
    def _design_matrix_for_genome(
        genome: Sequence[Mapping[str, Any]],
        X: np.ndarray,
        *,
        graph_cache: ExpressionGraphCache | None = None,
        batch_key: str | None = None,
    ) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if x.ndim != 2:
            raise ValueError("X must be 2D")
        if len(genome) <= 0:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        if graph_cache is None:
            phi = evaluate_genome_numpy(genome, x)
            return np.asarray(phi, dtype=float)
        cols: list[np.ndarray] = []
        for term in genome:
            expr = term.get("expr", term)
            z = graph_cache.evaluate_expression(
                expr,
                x,
                param_values=None,
                eps=1e-6,
                batch_key=batch_key,
            )
            cols.append(np.asarray(z, dtype=float).reshape(-1, 1))
        if not cols:
            return np.zeros((int(x.shape[0]), 0), dtype=float)
        return np.concatenate(cols, axis=1)

    @classmethod
    def _batched_ridge_predict(
        cls,
        *,
        genomes: Sequence[Sequence[Mapping[str, Any]]],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        l2_values: Sequence[float],
        graph_cache: ExpressionGraphCache | None = None,
        batch_key_train: str | None = None,
        batch_key_eval: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        xtr = np.asarray(X_train, dtype=float)
        ytr = _as_2d(np.asarray(y_train, dtype=float))
        xev = np.asarray(X_eval, dtype=float)
        B = int(len(genomes))
        if B <= 0:
            return np.zeros((0, int(xev.shape[0]), int(ytr.shape[1])), dtype=float), np.zeros((0, int(xtr.shape[0]), int(ytr.shape[1])), dtype=float)

        groups: dict[int, list[int]] = {}
        for i, g in enumerate(genomes):
            groups.setdefault(int(len(g)), []).append(int(i))

        pred_eval = np.zeros((B, int(xev.shape[0]), int(ytr.shape[1])), dtype=float)
        pred_train = np.zeros((B, int(xtr.shape[0]), int(ytr.shape[1])), dtype=float)

        try:
            import torch
        except Exception:
            # fallback to per-candidate ridge
            for i, g in enumerate(genomes):
                fit = evaluate_genome_with_ridge(
                    g,
                    X_train=xtr,
                    y_train=ytr,
                    X_eval=xev,
                    y_eval=None,
                    l2=float(max(0.0, l2_values[i])),
                )
                pred_eval[i] = _as_2d(np.asarray(fit.get("pred_eval"), dtype=float))
                pred_train[i] = _as_2d(np.asarray(fit.get("pred_train"), dtype=float))
            return pred_eval, pred_train

        ytr_t = torch.as_tensor(ytr, dtype=torch.float64)

        for k, idxs in groups.items():
            if int(k) <= 0:
                # intercept-only model
                b = np.mean(ytr, axis=0, keepdims=True)
                for i in idxs:
                    pred_train[i] = np.repeat(b, repeats=int(xtr.shape[0]), axis=0)
                    pred_eval[i] = np.repeat(b, repeats=int(xev.shape[0]), axis=0)
                continue

            phis_tr = []
            phis_ev = []
            reg_vals = []
            for i in idxs:
                g = genomes[int(i)]
                phis_tr.append(
                    cls._design_matrix_for_genome(
                        g,
                        xtr,
                        graph_cache=graph_cache,
                        batch_key=batch_key_train,
                    )
                )
                phis_ev.append(
                    cls._design_matrix_for_genome(
                        g,
                        xev,
                        graph_cache=graph_cache,
                        batch_key=batch_key_eval,
                    )
                )
                reg_vals.append(float(max(0.0, l2_values[int(i)])))

            A_tr = np.asarray(np.stack(phis_tr, axis=0), dtype=float)  # [Bg, n, k]
            A_ev = np.asarray(np.stack(phis_ev, axis=0), dtype=float)
            Bg = int(A_tr.shape[0])
            ones_tr = np.ones((Bg, int(A_tr.shape[1]), 1), dtype=float)
            ones_ev = np.ones((Bg, int(A_ev.shape[1]), 1), dtype=float)
            Atr = np.concatenate([A_tr, ones_tr], axis=2)  # [Bg,n,k+1]
            Aev = np.concatenate([A_ev, ones_ev], axis=2)

            Atr_t = torch.as_tensor(Atr, dtype=torch.float64)
            Aev_t = torch.as_tensor(Aev, dtype=torch.float64)
            yb_t = ytr_t.unsqueeze(0).expand(Bg, -1, -1)  # [Bg,n,m]

            At = Atr_t.transpose(1, 2)  # [Bg,k+1,n]
            lhs = torch.bmm(At, Atr_t)  # [Bg,k+1,k+1]
            rhs = torch.bmm(At, yb_t)  # [Bg,k+1,m]

            reg = torch.eye(int(k + 1), dtype=torch.float64).unsqueeze(0).repeat(Bg, 1, 1)
            reg[:, -1, -1] = 0.0
            lam = torch.as_tensor(np.asarray(reg_vals, dtype=float), dtype=torch.float64).reshape(Bg, 1, 1)
            lhs = lhs + lam * reg

            try:
                W = torch.linalg.solve(lhs, rhs)  # [Bg,k+1,m]
            except Exception:
                W = torch.matmul(torch.linalg.pinv(lhs), rhs)

            pred_tr_g = torch.bmm(Atr_t, W).cpu().numpy()
            pred_ev_g = torch.bmm(Aev_t, W).cpu().numpy()
            for loc, i in enumerate(idxs):
                pred_train[int(i)] = np.asarray(pred_tr_g[int(loc)], dtype=float)
                pred_eval[int(i)] = np.asarray(pred_ev_g[int(loc)], dtype=float)

        return pred_eval, pred_train

    def evaluate_population_batch(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pop = np.asarray(population, dtype=float)
        if pop.ndim == 1:
            pop = pop.reshape(1, -1)
        n = int(pop.shape[0])
        out_obj = np.zeros((n, 3), dtype=float)
        out_vio = np.zeros((n,), dtype=float)
        if n <= 0:
            return out_obj, out_vio

        decoded: list[tuple[list[int], int, dict[str, Any]]] = []
        cache_keys: list[tuple[tuple[int, ...], str]] = []
        need_eval_idx: list[int] = []
        for i in range(n):
            subset_idx, _, meta = self._decode(np.asarray(pop[i], dtype=float))
            key_subset = tuple(sorted(int(v) for v in subset_idx))
            key = (key_subset, self._cache_sig(meta))
            decoded.append((subset_idx, int(len(subset_idx)), meta))
            cache_keys.append(key)
            if key in self._cache:
                obj, _detail = self._cache[key]
                out_obj[i] = np.asarray(obj, dtype=float)
            else:
                need_eval_idx.append(int(i))

        if not need_eval_idx:
            return out_obj, out_vio

        genomes: list[list[Mapping[str, Any]]] = []
        metas: list[Mapping[str, Any]] = []
        for i in need_eval_idx:
            subset_idx, _k, meta = decoded[i]
            g = [{"name": self.candidates[j].name, "expr": dict(self.candidates[j].expr)} for j in sorted(int(v) for v in subset_idx)]
            genomes.append(g)
            metas.append(meta)

        B = int(len(genomes))
        fold_rmse = [[] for _ in range(B)]
        fold_branch = [[] for _ in range(B)]

        for fold_id, (tr_idx, va_idx) in enumerate(self.splits):
            tr_arr = np.asarray(tr_idx, dtype=int)
            va_arr = np.asarray(va_idx, dtype=int)
            Xtr = np.asarray(self.X_fit[tr_arr], dtype=float)
            ytr = np.asarray(self.y_fit[tr_arr], dtype=float)
            Xva = np.asarray(self.X_fit[va_arr], dtype=float)
            yva = np.asarray(self.y_fit[va_arr], dtype=float)

            l2s = [float(max(0.0, m.get("tuned_l2", self.base_ridge_l2))) for m in metas]
            pred_global, _pred_train = self._batched_ridge_predict(
                genomes=genomes,
                X_train=Xtr,
                y_train=ytr,
                X_eval=Xva,
                l2_values=l2s,
                graph_cache=self.graph_cache,
                batch_key_train=f"fold{int(fold_id)}|global|tr",
                batch_key_eval=f"fold{int(fold_id)}|global|va",
            )  # [B,nva,m]

            if not self.strict4_branch_mode or self.strict4_gate_idx is None:
                for bi in range(B):
                    rm = float(_rmse(yva, pred_global[bi]))
                    fold_rmse[bi].append(rm)
                    fold_branch[bi].append({})
                continue

            keys_tr = _strict4_keys_from_X(Xtr, self.strict4_gate_idx)
            keys_va = _strict4_keys_from_X(Xva, self.strict4_gate_idx)
            idx_tr_by_key = {k: np.asarray([ii for ii, kk in enumerate(keys_tr) if kk == k], dtype=int) for k in STRICT4_REGIME_ORDER}
            idx_va_by_key = {k: np.asarray([ii for ii, kk in enumerate(keys_va) if kk == k], dtype=int) for k in STRICT4_REGIME_ORDER}

            pred_va = np.asarray(pred_global, dtype=float).copy()  # [B,nva,m]
            branch_detail_all: list[dict[str, Any]] = [{"branch_rmse": {}, "branch_train_size": {}, "branch_fallback": {}} for _ in range(B)]

            for regime in STRICT4_REGIME_ORDER:
                tr_local = np.asarray(idx_tr_by_key[regime], dtype=int)
                va_local = np.asarray(idx_va_by_key[regime], dtype=int)
                if int(va_local.size) <= 0:
                    continue

                active_local: list[int] = []
                for bi, meta in enumerate(metas):
                    strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))
                    min_train = int(max(self.base_strict4_min_branch_train, round(strict4_ratio * float(tr_arr.size))))
                    use_branch = int(tr_local.size) >= int(min_train)
                    branch_detail_all[bi]["branch_train_size"][str(regime)] = int(tr_local.size)
                    branch_detail_all[bi]["branch_fallback"][str(regime)] = bool(not use_branch)
                    if use_branch:
                        active_local.append(int(bi))

                if active_local:
                    genomes_act = [genomes[bi] for bi in active_local]
                    l2s_act = [l2s[bi] for bi in active_local]
                    pred_loc, _ = self._batched_ridge_predict(
                        genomes=genomes_act,
                        X_train=Xtr[tr_local],
                        y_train=ytr[tr_local],
                        X_eval=Xva[va_local],
                        l2_values=l2s_act,
                        graph_cache=self.graph_cache,
                        batch_key_train=f"fold{int(fold_id)}|{str(regime)}|tr",
                        batch_key_eval=f"fold{int(fold_id)}|{str(regime)}|va",
                    )
                    for kpos, bi in enumerate(active_local):
                        pred_va[bi, va_local, :] = pred_loc[kpos]

                for bi in range(B):
                    yk = np.asarray(yva[va_local], dtype=float).reshape(-1)
                    pk = np.asarray(pred_va[bi, va_local, :], dtype=float).reshape(-1)
                    branch_detail_all[bi]["branch_rmse"][str(regime)] = float(_rmse(yk, pk))

            for bi in range(B):
                rm = float(_rmse(yva, pred_va[bi]))
                fold_rmse[bi].append(rm)
                fold_branch[bi].append(dict(branch_detail_all[bi]))

        # finalize and write cache
        for loc, i in enumerate(need_eval_idx):
            subset_idx, _k, meta = decoded[i]
            key_subset = tuple(sorted(int(v) for v in subset_idx))
            key = (key_subset, self._cache_sig(meta))
            key_int = [int(v) for v in key_subset]
            rm_arr = np.asarray(fold_rmse[loc], dtype=float)
            rmse_mean = float(np.mean(rm_arr))
            rmse_std = float(np.std(rm_arr))
            rmse_drift = float(np.mean(np.abs(np.diff(rm_arr)))) if rm_arr.size >= 2 else 0.0
            complexity = float(sum(float(self.candidates[j].complexity) for j in key_int))

            fam_counts: dict[str, int] = {}
            feat_counts: dict[int, int] = {}
            for j in key_int:
                c = self.candidates[j]
                fam_counts[str(c.family)] = int(fam_counts.get(str(c.family), 0) + 1)
                for f in c.features:
                    feat_counts[int(f)] = int(feat_counts.get(int(f), 0) + 1)
            fam_share = np.asarray([float(v) for v in fam_counts.values()], dtype=float)
            if fam_share.size > 0:
                fam_share = fam_share / float(np.sum(fam_share))
            feat_share = np.asarray([float(v) for v in feat_counts.values()], dtype=float)
            if feat_share.size > 0:
                feat_share = feat_share / float(np.sum(feat_share))
            fam_concentration = float(np.sum(fam_share**2)) if fam_share.size > 0 else 1.0
            feat_concentration = float(np.sum(feat_share**2)) if feat_share.size > 0 else 1.0

            complexity_scale = float(max(0.05, meta.get("complexity_scale", 1.0)))
            family_penalty_scale = float(max(0.05, meta.get("family_penalty_scale", 1.0)))
            feature_penalty_scale = float(max(0.05, meta.get("feature_penalty_scale", 1.0)))
            drift_weight = float(max(0.0, meta.get("drift_weight", 0.15)))
            tuned_l2 = float(max(0.0, meta.get("tuned_l2", self.base_ridge_l2)))
            strict4_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))

            obj = np.asarray(
                [
                    float(rmse_mean),
                    float(rmse_std + drift_weight * rmse_drift),
                    float(
                        complexity_scale * (complexity / max(1.0, float(self.max_terms)))
                        + family_penalty_scale * fam_concentration
                        + feature_penalty_scale * feat_concentration
                    ),
                ],
                dtype=float,
            )
            detail = {
                "subset_size": int(len(key_int)),
                "subset_idx": [int(v) for v in key_int],
                "subset_names": [self.candidates[j].name for j in key_int],
                "subset_families": [self.candidates[j].family for j in key_int],
                "fold_rmse": [float(v) for v in rm_arr.tolist()],
                "fold_branch_detail": _jsonable(fold_branch[loc]),
                "rmse_mean": float(rmse_mean),
                "rmse_std": float(rmse_std),
                "rmse_drift": float(rmse_drift),
                "complexity_raw": float(complexity),
                "family_concentration": float(fam_concentration),
                "feature_concentration": float(feat_concentration),
                "tuned_l2": float(tuned_l2),
                "strict4_min_train_ratio": float(strict4_ratio),
                "complexity_scale": float(complexity_scale),
                "family_penalty_scale": float(family_penalty_scale),
                "feature_penalty_scale": float(feature_penalty_scale),
                "drift_weight": float(drift_weight),
                "decode_meta": _jsonable(dict(meta)),
            }
            self._cache[key] = (obj, detail)
            out_obj[i] = np.asarray(obj, dtype=float)

        return out_obj, out_vio

    def evaluate(self, x):
        try:
            subset_idx, _, meta = self._decode(np.asarray(x, dtype=float))
            obj, _ = self._evaluate_subset(subset_idx, meta)
            return np.asarray(obj, dtype=float)
        except Exception:
            return np.asarray([1e6, 1e3, 1e3], dtype=float)

    def cache_top(self, *, topn: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, (obj, detail) in self._cache.items():
            rows.append(
                {
                    "obj_accuracy": float(obj[0]),
                    "obj_stability": float(obj[1]),
                    "obj_complexity": float(obj[2]),
                    **_jsonable(detail),
                }
            )
        rows.sort(
            key=lambda r: (
                float(r["obj_accuracy"]),
                float(r["obj_stability"]),
                float(r["obj_complexity"]),
            )
        )
        return rows[: int(max(1, topn))]

__all__ = ['SymbolicSubsetSelectionProblem']

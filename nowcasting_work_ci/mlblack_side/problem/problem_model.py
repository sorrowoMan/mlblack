from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from bias import ObjectivePolicyConfig, build_interval_row_objective_key
from core.symbolic.expression_graph_cache import ExpressionGraphCache
from core.symbolic.feature_space.regime_router import RegimePolicy
from evaluation import (
    FitPredictCallbackConfig,
    IntervalCallbackConfig,
    ProblemEvaluationCallbacks,
)
from core.execution import ExecutionResourceGrant, ExecutionResourceRequest
from model import _jsonable, _rmse
from nowcasting_work_ci.mlblack_side.problem.domain_router import WORK_CI_STRICT4_POLICY
from pipeline.feature_space import (
    CandidateTerm,
    DEFAULT_META_SIGNATURE_KEYS,
    batched_ridge_predict,
    build_interval_subset_report,
    build_rolling_splits,
    build_subset_meta_cache_key,
    build_subset_descriptor,
    interval_metrics_batch,
    symmetric_interval_batch,
    evaluate_regime_fold,
    evaluate_global_fold,
    evaluate_symmetric_residual_fold_batch,
)
from problem import DecisionEvaluationBridge, DecodedDecision
from nsgablack.core.base import BlackBoxProblem
from training.inner_runtime import InnerRuntimeDispatcher, InnerRuntimeHook



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
        random_seed: int | None,
        regime_branch_mode: bool | None = None,
        regime_gate_idx: tuple[int, int, int, int] | None = None,
        regime_min_branch_train: int | None = None,
        regime_branch_parallel_workers: int | None = None,
        regime_policy: RegimePolicy | None = None,
        strict4_branch_mode: bool | None = None,
        strict4_gate_idx: tuple[int, int, int, int] | None = None,
        strict4_min_branch_train: int | None = None,
        strict4_branch_parallel_workers: int | None = None,
        strict4_router_spec: RegimePolicy | None = None,
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
        interval_alpha: float,
        interval_method: str,
        interval_calib_ratio: float,
        interval_quantile_l2: float,
        selection_coverage_error_threshold: float,
        graph_cache: ExpressionGraphCache | None = None,
        inner_runtime_hooks: Sequence[InnerRuntimeHook] = (),
        inner_runtime_context: Mapping[str, Any] | None = None,
        inner_runtime_strict: bool = False,
    ) -> None:
        self.X_fit = np.asarray(X_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        self.candidates = list(candidates)
        self.families = tuple(sorted({str(c.family) for c in self.candidates}))
        self.family_to_idx = {str(v): int(i) for i, v in enumerate(self.families)}
        self.max_terms = int(max(2, max_terms))
        self.base_ridge_l2 = float(max(0.0, ridge_l2))
        self.random_seed = None if random_seed is None else int(random_seed)
        self.rolling_folds = int(max(1, rolling_folds))
        self.rolling_val_ratio = float(rolling_val_ratio)
        self.min_train = int(max(128, min_train))
        self.regime_branch_mode = bool(
            regime_branch_mode if regime_branch_mode is not None else bool(strict4_branch_mode)
        )
        self.regime_gate_idx = regime_gate_idx if regime_gate_idx is not None else strict4_gate_idx
        self.regime_policy = (
            regime_policy
            if regime_policy is not None
            else strict4_router_spec
            if strict4_router_spec is not None
            else WORK_CI_STRICT4_POLICY
        )
        _resolved_regime_min_branch_train = (
            regime_min_branch_train if regime_min_branch_train is not None else strict4_min_branch_train
        )
        _resolved_regime_branch_parallel_workers = (
            regime_branch_parallel_workers
            if regime_branch_parallel_workers is not None
            else strict4_branch_parallel_workers
        )
        if _resolved_regime_min_branch_train is None:
            raise ValueError("regime_min_branch_train is required")
        if _resolved_regime_branch_parallel_workers is None:
            raise ValueError("regime_branch_parallel_workers is required")
        self.base_regime_min_branch_train = int(max(8, _resolved_regime_min_branch_train))
        self.regime_branch_parallel_workers = int(max(1, _resolved_regime_branch_parallel_workers))
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
        self.interval_alpha = float(np.clip(interval_alpha, 1e-6, 0.99))
        self.interval_method = str(interval_method)
        self.interval_calib_ratio = float(np.clip(interval_calib_ratio, 0.05, 0.4))
        self.interval_quantile_l2 = float(max(0.0, interval_quantile_l2))
        self.selection_coverage_error_threshold = float(max(0.0, selection_coverage_error_threshold))
        self.objective_policy = ObjectivePolicyConfig(
            coverage_error_threshold=float(self.selection_coverage_error_threshold),
        )
        self.graph_cache = graph_cache
        inner_dispatcher = (
            InnerRuntimeDispatcher.from_hooks(inner_runtime_hooks, strict=bool(inner_runtime_strict))
            if tuple(inner_runtime_hooks)
            else None
        )
        base_inner_runtime_context = {} if inner_runtime_context is None else dict(inner_runtime_context)
        base_inner_runtime_context.setdefault("task_id", "symbolic_subset_selection_problem")
        base_inner_runtime_context.setdefault("run_id", str(base_inner_runtime_context.get("task_id")))
        base_inner_runtime_context.setdefault("trainer_name", "symbolic_subset_selection_problem")
        self._callbacks = ProblemEvaluationCallbacks(
            interval_config=IntervalCallbackConfig(
                interval_alpha=float(self.interval_alpha),
                interval_method=str(self.interval_method),
                interval_calib_ratio=float(self.interval_calib_ratio),
                interval_quantile_l2=float(self.interval_quantile_l2),
                regime_branch_mode=bool(self.regime_branch_mode),
                regime_gate_idx=self.regime_gate_idx,
                base_regime_min_branch_train=int(self.base_regime_min_branch_train),
                regime_branch_parallel_workers=int(self.regime_branch_parallel_workers),
                regime_policy=self.regime_policy,
            ),
            fit_predict_config=FitPredictCallbackConfig(
                random_seed=self.random_seed,
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
            ),
            jsonable_fn=_jsonable,
            rmse_fn=_rmse,
            inner_runtime_dispatcher=inner_dispatcher,
            inner_runtime_context=base_inner_runtime_context,
        )
        self.splits = build_rolling_splits(
            int(self.X_fit.shape[0]),
            folds=int(self.rolling_folds),
            val_ratio=float(self.rolling_val_ratio),
            min_train=int(self.min_train),
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
        self._eval_bridge = DecisionEvaluationBridge(
            decode_fn=self._decode,
            evaluate_decoded_fn=self._evaluate_decoded,
            evaluate_decoded_batch_fn=self._evaluate_decoded_batch,
            objective_dim=3,
            fallback_objectives=(1e6, 1e3, 1e3),
        )

    @property
    def strict4_branch_mode(self) -> bool:
        return self.regime_branch_mode

    @property
    def strict4_gate_idx(self) -> tuple[int, int, int, int] | None:
        return self.regime_gate_idx

    @property
    def strict4_router_spec(self) -> RegimePolicy:
        return self.regime_policy

    @property
    def base_strict4_min_branch_train(self) -> int:
        return self.base_regime_min_branch_train

    @property
    def strict4_branch_parallel_workers(self) -> int:
        return self.regime_branch_parallel_workers

    def execution_resource_requests(self) -> tuple[ExecutionResourceRequest, ...]:
        return tuple(
            self._callbacks.execution_resource_requests(
                rolling_folds=int(self.rolling_folds),
                label="symbolic_subset_selection_problem",
            )
        )

    def execution_resource_request(self) -> ExecutionResourceRequest:
        request = self._callbacks.execution_resource_request(
            rolling_folds=int(self.rolling_folds),
            label="symbolic_subset_selection_problem",
        )
        metadata = dict(request.metadata)
        metadata.update(
            {
                "rolling_val_ratio": float(self.rolling_val_ratio),
                "min_train": int(self.min_train),
                "objective_dim": 3,
            }
        )
        return ExecutionResourceRequest(
            threads=int(request.threads),
            backend=str(request.backend),
            label=str(request.label),
            device_tokens=tuple(request.device_tokens),
            metadata=metadata,
        )

    def execution_resource_grant(self) -> ExecutionResourceGrant | None:
        return self._callbacks.execution_resource_grant()

    def set_execution_resource_grant(
        self,
        grant: ExecutionResourceGrant | ExecutionResourceRequest | Mapping[str, Any] | None,
    ) -> ExecutionResourceGrant | None:
        return self._callbacks.set_execution_resource_grant(grant)

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

    def _eval_fold_global(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
        fold_id: int | None = None,
    ) -> dict[str, Any]:
        return evaluate_global_fold(
            genome=genome,
            X_fit=self.X_fit,
            y_fit=self.y_fit,
            tr_idx=tr_idx,
            va_idx=va_idx,
            l2=l2,
            fit_predict_fn=self._callbacks.fit_predict,
            build_interval_bounds_fn=self._callbacks.build_interval_bounds,
            summarize_fold_fn=self._callbacks.summarize_fold,
            inner_runtime_dispatcher=self._callbacks.inner_runtime_dispatcher,
            inner_runtime_context=self._callbacks.build_inner_runtime_context(
                {
                    "run_suffix": ("fold_global" if fold_id is None else f"fold_global:{int(fold_id)}"),
                    "fold_id": None if fold_id is None else int(fold_id),
                }
            ),
        )

    def _eval_fold_regime(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
        regime_min_branch_train: int,
        fold_id: int | None = None,
    ) -> dict[str, Any]:
        return evaluate_regime_fold(
            genome=genome,
            X_fit=self.X_fit,
            y_fit=self.y_fit,
            tr_idx=tr_idx,
            va_idx=va_idx,
            l2=l2,
            regime_min_branch_train=regime_min_branch_train,
            config=self._callbacks.branch_eval_config(),
            fit_predict_fn=self._callbacks.fit_predict,
            build_interval_bounds_fn=self._callbacks.build_interval_bounds,
            summarize_fold_fn=self._callbacks.summarize_fold,
            inner_runtime_dispatcher=self._callbacks.inner_runtime_dispatcher,
            inner_runtime_context=self._callbacks.build_inner_runtime_context(
                {
                    "run_suffix": ("fold_regime" if fold_id is None else f"fold_regime:{int(fold_id)}"),
                    "fold_id": None if fold_id is None else int(fold_id),
                }
            ),
        )

    def _eval_fold_strict4(
        self,
        genome: Sequence[Mapping[str, Any]],
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
        *,
        l2: float,
        strict4_min_branch_train: int,
    ) -> dict[str, Any]:
        return self._eval_fold_regime(
            genome,
            tr_idx=tr_idx,
            va_idx=va_idx,
            l2=l2,
            regime_min_branch_train=strict4_min_branch_train,
        )

    def _evaluate_subset(self, subset_idx: Sequence[int], meta: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        key = build_subset_meta_cache_key(
            subset_idx,
            meta,
            keys=DEFAULT_META_SIGNATURE_KEYS,
            float_precision=7,
        )
        key_subset = key[0]
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        subset_descriptor = build_subset_descriptor(candidates=self.candidates, subset_idx=key_subset)
        genome = list(subset_descriptor["genome"])
        tuned_l2 = float(max(0.0, meta.get("tuned_l2", self.base_ridge_l2)))
        regime_min_train_ratio = float(np.clip(meta.get("strict4_min_train_ratio", 0.08), 0.01, 0.30))

        fold_results: list[dict[str, Any]] = []
        for fold_id, (tr_idx, va_idx) in enumerate(self.splits):
            tr_arr = np.asarray(tr_idx, dtype=int)
            va_arr = np.asarray(va_idx, dtype=int)
            regime_min_train = int(
                max(
                    self.base_regime_min_branch_train,
                    round(float(regime_min_train_ratio) * float(tr_arr.size)),
                )
            )
            fold_res = self._eval_fold_regime(
                genome,
                tr_idx=tr_arr,
                va_idx=va_arr,
                l2=tuned_l2,
                regime_min_branch_train=regime_min_train,
                fold_id=int(fold_id),
            )
            fold_results.append(dict(fold_res))

        out_obj, detail = build_interval_subset_report(
            subset_idx=key_subset,
            subset_candidates=list(subset_descriptor["subset_candidates"]),
            fold_results=fold_results,
            decode_meta=dict(meta),
            selection_coverage_error_threshold=float(self.selection_coverage_error_threshold),
            jsonable_fn=_jsonable,
        )
        self._cache[key] = (out_obj, detail)
        return out_obj, detail

    def _evaluate_decoded(self, decoded: DecodedDecision) -> np.ndarray:
        obj, _ = self._evaluate_subset(decoded.subset_idx, decoded.meta)
        return np.asarray(obj, dtype=float)

    def _evaluate_decoded_batch(self, decoded_batch: Sequence[DecodedDecision]) -> np.ndarray:
        n = int(len(decoded_batch))
        out_obj = np.zeros((n, 3), dtype=float)
        if n <= 0:
            return out_obj

        decoded: list[DecodedDecision] = [d for d in decoded_batch]
        need_eval_idx: list[int] = []
        for i in range(n):
            meta = dict(decoded[i].meta)
            key = build_subset_meta_cache_key(
                decoded[i].subset_idx,
                meta,
                keys=DEFAULT_META_SIGNATURE_KEYS,
                float_precision=7,
            )
            if key in self._cache:
                obj, _detail = self._cache[key]
                out_obj[i] = np.asarray(obj, dtype=float)
            else:
                need_eval_idx.append(int(i))

        if not need_eval_idx:
            return out_obj

        if self.interval_method != "symmetric_residual":
            for i in need_eval_idx:
                obj, _detail = self._evaluate_subset(decoded[i].subset_idx, decoded[i].meta)
                out_obj[i] = np.asarray(obj, dtype=float)
            return out_obj

        genomes: list[list[Mapping[str, Any]]] = []
        metas: list[Mapping[str, Any]] = []
        for i in need_eval_idx:
            subset_idx = tuple(sorted(int(v) for v in decoded[i].subset_idx))
            meta = dict(decoded[i].meta)
            subset_descriptor = build_subset_descriptor(candidates=self.candidates, subset_idx=subset_idx)
            genomes.append(list(subset_descriptor["genome"]))
            metas.append(meta)

        B = int(len(genomes))
        fold_results_by_candidate = [[] for _ in range(B)]

        for fold_id, (tr_idx, va_idx) in enumerate(self.splits):
            tr_arr = np.asarray(tr_idx, dtype=int)
            va_arr = np.asarray(va_idx, dtype=int)
            fold_results = evaluate_symmetric_residual_fold_batch(
                genomes=genomes,
                metas=metas,
                X_fit=self.X_fit,
                y_fit=self.y_fit,
                tr_idx=tr_arr,
                va_idx=va_arr,
                base_ridge_l2=float(self.base_ridge_l2),
                config=self._callbacks.branch_eval_config(),
                batched_predict_fn=batched_ridge_predict,
                symmetric_interval_batch_fn=symmetric_interval_batch,
                interval_metrics_batch_fn=interval_metrics_batch,
                summarize_fold_fn=self._callbacks.summarize_fold,
                rmse_fn=_rmse,
                graph_cache=self.graph_cache,
                batch_key_prefix=f"fold{int(fold_id)}",
                inner_runtime_dispatcher=self._callbacks.inner_runtime_dispatcher,
                inner_runtime_context=self._callbacks.build_inner_runtime_context(
                    {
                        "run_suffix": f"fold_batch:{int(fold_id)}",
                        "fold_id": int(fold_id),
                    }
                ),
            )
            for bi, fold_res in enumerate(fold_results):
                fold_results_by_candidate[bi].append(dict(fold_res))

        # finalize and write cache
        for loc, i in enumerate(need_eval_idx):
            meta = dict(decoded[i].meta)
            key = build_subset_meta_cache_key(
                decoded[i].subset_idx,
                meta,
                keys=DEFAULT_META_SIGNATURE_KEYS,
                float_precision=7,
            )
            key_subset = key[0]
            subset_descriptor = build_subset_descriptor(candidates=self.candidates, subset_idx=key_subset)
            obj, detail = build_interval_subset_report(
                subset_idx=key_subset,
                subset_candidates=list(subset_descriptor["subset_candidates"]),
                fold_results=fold_results_by_candidate[loc],
                decode_meta=meta,
                selection_coverage_error_threshold=float(self.selection_coverage_error_threshold),
                jsonable_fn=_jsonable,
            )
            self._cache[key] = (obj, detail)
            out_obj[i] = np.asarray(obj, dtype=float)

        return out_obj

    def evaluate_population_batch(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._eval_bridge.evaluate_population(np.asarray(population, dtype=float))

    def evaluate(self, x):
        return self._eval_bridge.evaluate_one(np.asarray(x, dtype=float))

    def cache_top(self, *, topn: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, (obj, detail) in self._cache.items():
            rows.append(
                {
                    "obj_coverage_error": float(obj[0]),
                    "obj_pinaw": float(obj[1]),
                    "obj_interval_score": float(obj[2]),
                    **_jsonable(detail),
                }
            )
        rows.sort(
            key=lambda r: build_interval_row_objective_key(r, cfg=self.objective_policy)
        )
        return rows[: int(max(1, topn))]



from __future__ import annotations

from .legacy_imports import *
from ..artifacts.serialization import *
from ..evaluation.metrics import *
from ..pipeline.inner_fit import *
from ..pipeline.splits import *
from ..pipeline.regimes import *
from ..pipeline.candidate_pool import *
from ..pipeline.main import build_pipeline
from ..adapter.outer_adapter import *
from ..problem.symbolic_subset_problem import SymbolicSubsetSelectionProblem

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NOWCASTING package: NSGABLACK outer (subset optimization) + MLBLACK inner (symbolic ridge eval) on Work-CI (same-day state estimation, not strict t+1 forecasting)."
    )
    parser.add_argument("--csv-path", type=str, default=default_work_ci_csv())
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--rolling-val-ratio", type=float, default=0.18)
    parser.add_argument("--max-terms", type=int, default=12)
    parser.add_argument("--ridge-l2", type=float, default=1e-4)
    parser.add_argument("--strict4-branch-mode", action="store_true")
    parser.add_argument("--strict4-min-branch-train", type=int, default=64)
    parser.add_argument("--strict4-branch-parallel-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outer-strategy", type=str, default="portfolio", choices=["nsga2", "moead", "vns", "portfolio"])
    parser.add_argument("--portfolio-phases", type=str, default="nsga2,moead,vns")
    parser.add_argument("--portfolio-phase-weights", type=str, default="2,1,1")
    parser.add_argument("--moead-neighborhood-size", type=int, default=12)
    parser.add_argument("--moead-delta", type=float, default=0.9)
    parser.add_argument("--moead-nr", type=int, default=2)
    parser.add_argument("--vns-k-max", type=int, default=5)
    parser.add_argument("--vns-batch-size", type=int, default=32)
    parser.add_argument("--inner-opt-enabled", type=int, default=1)
    parser.add_argument("--inner-opt-adam-steps", type=int, default=80)
    parser.add_argument("--inner-opt-adam-lr", type=float, default=1e-2)
    parser.add_argument("--inner-opt-lbfgs-steps", type=int, default=25)
    parser.add_argument("--inner-opt-lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--inner-opt-accept-rmse-tol", type=float, default=0.0)
    parser.add_argument("--inner-opt-accept-rel-tol", type=float, default=0.01)
    parser.add_argument("--inner-opt-guard-patience", type=int, default=3)
    parser.add_argument("--inner-opt-guard-check-interval", type=int, default=10)
    parser.add_argument("--inner-opt-alt-freeze-readout", type=int, default=1)
    parser.add_argument("--inner-opt-grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--inner-opt-residual-clip-q", type=float, default=0.98)
    parser.add_argument("--batched-eval", type=int, default=1)
    parser.add_argument("--reinvest-search", type=int, default=1)
    parser.add_argument("--reinvest-pop-mult", type=float, default=1.5)
    parser.add_argument("--reinvest-gen-mult", type=float, default=1.5)
    parser.add_argument("--reinvest-strict4-workers-mult", type=float, default=1.5)
    parser.add_argument("--dynamic-pool-enabled", type=int, default=1)
    parser.add_argument("--dynamic-pool-epochs", type=int, default=4)
    parser.add_argument("--dynamic-init-minimal", type=int, default=1)
    parser.add_argument("--dynamic-expand-max-new", type=int, default=24)
    parser.add_argument("--dynamic-focus-top-features", type=int, default=5)
    parser.add_argument("--dynamic-partner-topk", type=int, default=4)
    parser.add_argument("--dynamic-top-cache-use", type=int, default=20)
    parser.add_argument("--dynamic-max-pool-size", type=int, default=240)
    parser.add_argument("--graph-cache-enabled", type=int, default=1)
    parser.add_argument("--graph-cache-backend", type=str, default="sqlite", choices=["memory", "sqlite"])
    parser.add_argument("--graph-cache-db-path", type=str, default="")
    parser.add_argument("--graph-cache-namespace", type=str, default="work_ci_nowcasting_subset_bridge")
    parser.add_argument("--graph-cache-persist-values", type=int, default=0)
    parser.add_argument("--interval-alpha", type=float, default=0.1, help="Interval alpha (target coverage=1-alpha).")
    parser.add_argument(
        "--interval-method",
        type=str,
        default="native_quantile_cqr",
        choices=["native_quantile_cqr", "symmetric_residual"],
        help="Interval construction method for symbolic model.",
    )
    parser.add_argument("--interval-calib-ratio", type=float, default=0.2, help="Calibration split ratio for conformal quantile interval.")
    parser.add_argument("--interval-quantile-l2", type=float, default=1e-4, help="L2 regularization for QuantileRegressor heads.")
    parser.add_argument("--safe-log1p-abs", type=int, default=1, help="Enable safe basis log(1+abs(x)).")
    parser.add_argument("--safe-exp-clip", type=int, default=1, help="Enable safe basis exp(clip-scaled x).")
    parser.add_argument("--safe-reciprocal", type=int, default=1, help="Enable safe basis 1/(abs(x)+eps).")
    parser.add_argument("--safe-exp-clip-k", type=float, default=8.0, help="Scale k for exp_clip: exp(x/k).")
    parser.add_argument("--safe-reciprocal-eps", type=float, default=1e-3, help="Epsilon for reciprocal_safe.")
    parser.add_argument("--lag-feature-enabled", type=int, default=1, help="If 1, generate lag features for selected sources.")
    parser.add_argument("--lag-orders", type=str, default="1,2,3", help="Comma-separated lag orders, e.g. 1,2,3")
    parser.add_argument(
        "--lag-sources",
        type=str,
        default="ci,total_flow,avg_speed,avg_occ",
        help="Comma-separated lag sources from {ci,total_flow,avg_speed,avg_occ}",
    )
    parser.add_argument(
        "--lag-cross-enabled",
        type=int,
        default=1,
        help="If 1, add hinge(ci_lag1,c)*avg_speed_lag1 cross-lag terms.",
    )
    parser.add_argument(
        "--lag-cross-quantiles",
        type=str,
        default="0.25,0.5,0.75",
        help="Quantiles for c in hinge(ci_lag1,c)*avg_speed_lag1",
    )
    parser.add_argument(
        "--drop-same-day-flow-speed-occ",
        type=int,
        default=1,
        help="If 1, drop leak-prone same-day features: total_flow/avg_speed/avg_occ.",
    )
    parser.add_argument(
        "--drop-feature-list",
        type=str,
        default="total_flow,avg_speed,avg_occ",
        help="Comma-separated feature names to drop from train/test feature matrices.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_root = PROJECT_ROOT / "out" / "symbolic_interval_outer" / f"nowcasting_symbolic_subset_bridge_work_ci_seed{int(args.seed)}_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    reader = WorkCiIntervalReader(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
    )
    bundle = reader.read()
    tr = bundle.train
    te = bundle.test
    if te is None:
        raise ValueError("no test split in reader output")

    X_train = np.asarray(tr.X_train, dtype=float)
    y_train = np.asarray(tr.y_train, dtype=float).reshape(-1, 1)
    X_test = np.asarray(te.X_train, dtype=float)
    y_test = np.asarray(te.y_train, dtype=float).reshape(-1, 1)
    feature_names = tuple(str(v) for v in tr.feature_names)
    n_features_raw = int(X_train.shape[1])
    feature_names_raw = tuple(str(v) for v in feature_names)
    lag_added_features: list[str] = []
    lag_cross_added_features: list[str] = []

    lag_enabled = bool(int(args.lag_feature_enabled))
    lag_orders = _parse_int_list_csv(str(args.lag_orders), default=(1, 2, 3))
    lag_sources = [s.strip() for s in str(args.lag_sources).split(",") if s.strip()]
    lag_source_set = set(lag_sources)
    valid_sources = {"ci", "total_flow", "avg_speed", "avg_occ"}
    lag_source_set = {s for s in lag_source_set if s in valid_sources}
    if lag_enabled and lag_orders and lag_source_set:
        name_to_idx_raw = {str(nm): int(i) for i, nm in enumerate(feature_names)}
        tr_cols: list[np.ndarray] = [np.asarray(X_train, dtype=float)]
        te_cols: list[np.ndarray] = [np.asarray(X_test, dtype=float)]
        ext_names: list[str] = list(feature_names)

        def _append_lags(src_name: str, tr_src: np.ndarray, te_src: np.ndarray) -> None:
            nonlocal tr_cols, te_cols, ext_names, lag_added_features
            for lag in lag_orders:
                tr_l, te_l = _make_lag_from_history(tr_src, te_src, int(lag))
                nm = f"{src_name}_lag{int(lag)}"
                tr_cols.append(tr_l.reshape(-1, 1))
                te_cols.append(te_l.reshape(-1, 1))
                ext_names.append(nm)
                lag_added_features.append(nm)

        if "ci" in lag_source_set:
            _append_lags("ci", y_train.reshape(-1), y_test.reshape(-1))
        for src in ("total_flow", "avg_speed", "avg_occ"):
            if src not in lag_source_set:
                continue
            idx = name_to_idx_raw.get(src)
            if idx is None:
                continue
            _append_lags(src, X_train[:, int(idx)].reshape(-1), X_test[:, int(idx)].reshape(-1))

        X_train = np.concatenate(tr_cols, axis=1)
        X_test = np.concatenate(te_cols, axis=1)
        feature_names = tuple(str(v) for v in ext_names)

    lag_cross_enabled = bool(int(args.lag_cross_enabled))
    lag_cross_q = [float(np.clip(v, 0.01, 0.99)) for v in _parse_float_list_csv(str(args.lag_cross_quantiles), default=(0.25, 0.5, 0.75))]
    if lag_cross_enabled:
        name_to_idx = {str(nm): int(i) for i, nm in enumerate(feature_names)}
        i_ci = name_to_idx.get("ci_lag1")
        i_sp = name_to_idx.get("avg_speed_lag1")
        if i_ci is not None and i_sp is not None:
            ci_tr = np.asarray(X_train[:, int(i_ci)], dtype=float).reshape(-1)
            ci_te = np.asarray(X_test[:, int(i_ci)], dtype=float).reshape(-1)
            sp_tr = np.asarray(X_train[:, int(i_sp)], dtype=float).reshape(-1)
            sp_te = np.asarray(X_test[:, int(i_sp)], dtype=float).reshape(-1)
            tr_cols = [np.asarray(X_train, dtype=float)]
            te_cols = [np.asarray(X_test, dtype=float)]
            ext_names = list(feature_names)
            for qv in lag_cross_q:
                c = float(np.quantile(ci_tr, float(qv)))
                hz_tr = np.maximum(0.0, ci_tr - c) * sp_tr
                hz_te = np.maximum(0.0, ci_te - c) * sp_te
                nm = f"hinge_ci_lag1_q{int(round(qv * 100.0)):02d}_x_avg_speed_lag1"
                tr_cols.append(hz_tr.reshape(-1, 1))
                te_cols.append(hz_te.reshape(-1, 1))
                ext_names.append(nm)
                lag_cross_added_features.append(nm)
            X_train = np.concatenate(tr_cols, axis=1)
            X_test = np.concatenate(te_cols, axis=1)
            feature_names = tuple(str(v) for v in ext_names)

    dropped_features: list[str] = []
    if bool(int(args.drop_same_day_flow_speed_occ)):
        drop_set = {s.strip() for s in str(args.drop_feature_list).split(",") if s.strip()}
        if drop_set:
            keep_idx = [i for i, nm in enumerate(feature_names) if str(nm) not in drop_set]
            keep_set = set(keep_idx)
            dropped_features = [str(feature_names[i]) for i in range(len(feature_names)) if i not in keep_set]
            if not keep_idx:
                raise ValueError("all features were dropped; adjust --drop-feature-list")
            X_train = np.asarray(X_train[:, keep_idx], dtype=float)
            X_test = np.asarray(X_test[:, keep_idx], dtype=float)
            feature_names = tuple(str(feature_names[i]) for i in keep_idx)

    gate_names = (
        "is_holiday_day_or_window",
        "is_holiday_near",
        "is_holiday_mid",
        "is_nonwork_weekend",
    )
    gate_idx_list = [feature_names.index(nm) for nm in gate_names if nm in feature_names]
    strict4_gate_idx: tuple[int, int, int, int] | None = None
    strict4_enabled = bool(args.strict4_branch_mode)
    if strict4_enabled:
        if len(gate_idx_list) != 4:
            strict4_enabled = False
        else:
            strict4_gate_idx = (
                int(gate_idx_list[0]),
                int(gate_idx_list[1]),
                int(gate_idx_list[2]),
                int(gate_idx_list[3]),
            )

    batched_eval_enabled = bool(int(args.batched_eval))
    reinvest_enabled = bool(int(args.reinvest_search))
    effective_pop_size = int(max(4, int(args.pop_size)))
    effective_generations = int(max(1, int(args.generations)))
    effective_strict4_workers = int(max(1, int(args.strict4_branch_parallel_workers)))
    if batched_eval_enabled and reinvest_enabled:
        effective_pop_size = int(max(effective_pop_size, round(effective_pop_size * float(max(1.0, args.reinvest_pop_mult)))))
        effective_generations = int(
            max(effective_generations, round(effective_generations * float(max(1.0, args.reinvest_gen_mult))))
        )
        if strict4_enabled:
            effective_strict4_workers = int(
                max(
                    effective_strict4_workers,
                    round(effective_strict4_workers * float(max(1.0, args.reinvest_strict4_workers_mult))),
                )
            )
    effective_vns_batch_size = int(max(4, int(args.vns_batch_size), effective_pop_size))

    dynamic_pool_enabled = bool(int(args.dynamic_pool_enabled))
    dynamic_pool_epochs = int(max(1, args.dynamic_pool_epochs))
    dynamic_init_minimal = bool(int(args.dynamic_init_minimal))
    dynamic_expand_max_new = int(max(1, args.dynamic_expand_max_new))
    dynamic_focus_top_features = int(max(2, args.dynamic_focus_top_features))
    dynamic_partner_topk = int(max(2, args.dynamic_partner_topk))
    dynamic_top_cache_use = int(max(5, args.dynamic_top_cache_use))
    dynamic_max_pool_size = int(max(32, args.dynamic_max_pool_size))

    graph_cache_enabled = bool(int(args.graph_cache_enabled))
    graph_cache_backend = str(args.graph_cache_backend).strip().lower()
    graph_cache_db_path = str(args.graph_cache_db_path).strip()
    if graph_cache_enabled and graph_cache_backend == "sqlite" and not graph_cache_db_path:
        graph_cache_db_path = str((ROOT / ".mlblack_cache" / "work_ci_subset_expression_graph_cache.sqlite3"))
    graph_cache = ExpressionGraphCache(
        enabled=bool(graph_cache_enabled),
        backend=str(graph_cache_backend),
        db_path=str(graph_cache_db_path),
        namespace=str(args.graph_cache_namespace),
        persist_values=bool(int(args.graph_cache_persist_values)),
    )
    safe_log1p_abs_enabled = bool(int(args.safe_log1p_abs))
    safe_exp_clip_enabled = bool(int(args.safe_exp_clip))
    safe_reciprocal_enabled = bool(int(args.safe_reciprocal))
    safe_exp_clip_k = float(max(1.0, args.safe_exp_clip_k))
    safe_reciprocal_eps = float(max(1e-8, args.safe_reciprocal_eps))

    candidates = _build_candidate_pool(
        X_train,
        y_train,
        feature_names=feature_names,
        topk_for_pairs=6,
        include_pair_interactions=bool(not dynamic_init_minimal),
        include_gradient_enrich=bool(not dynamic_init_minimal),
        include_safe_log1p_abs=bool(safe_log1p_abs_enabled),
        include_safe_exp_clip=bool(safe_exp_clip_enabled),
        include_safe_reciprocal=bool(safe_reciprocal_enabled),
        safe_exp_clip_k=float(safe_exp_clip_k),
        safe_reciprocal_eps=float(safe_reciprocal_eps),
    )

    if not dynamic_pool_enabled:
        candidates = _build_candidate_pool(
            X_train,
            y_train,
            feature_names=feature_names,
            topk_for_pairs=6,
            include_pair_interactions=True,
            include_gradient_enrich=True,
            include_safe_log1p_abs=bool(safe_log1p_abs_enabled),
            include_safe_exp_clip=bool(safe_exp_clip_enabled),
            include_safe_reciprocal=bool(safe_reciprocal_enabled),
            safe_exp_clip_k=float(safe_exp_clip_k),
            safe_reciprocal_eps=float(safe_reciprocal_eps),
        )

    def _run_outer_once(
        *,
        run_candidates: Sequence[CandidateTerm],
        generations_this_epoch: int,
        seed_this_epoch: int,
    ) -> tuple[SymbolicSubsetSelectionProblem, dict[str, Any], dict[str, Any], float]:
        problem_local = SymbolicSubsetSelectionProblem(
            X_fit=X_train,
            y_fit=y_train,
            candidates=run_candidates,
            max_terms=int(max(2, args.max_terms)),
            ridge_l2=float(max(0.0, args.ridge_l2)),
            rolling_folds=int(max(1, args.rolling_folds)),
            rolling_val_ratio=float(np.clip(args.rolling_val_ratio, 0.05, 0.45)),
            min_train=max(256, int(round(0.4 * X_train.shape[0]))),
            strict4_branch_mode=bool(strict4_enabled),
            strict4_gate_idx=strict4_gate_idx,
            strict4_min_branch_train=int(max(8, args.strict4_min_branch_train)),
            strict4_branch_parallel_workers=int(effective_strict4_workers),
            inner_opt_enabled=bool(int(args.inner_opt_enabled)),
            inner_opt_adam_steps=int(max(0, args.inner_opt_adam_steps)),
            inner_opt_adam_lr=float(max(1e-8, args.inner_opt_adam_lr)),
            inner_opt_lbfgs_steps=int(max(0, args.inner_opt_lbfgs_steps)),
            inner_opt_lbfgs_lr=float(max(1e-8, args.inner_opt_lbfgs_lr)),
            inner_opt_accept_rmse_tol=float(max(0.0, args.inner_opt_accept_rmse_tol)),
            inner_opt_accept_rel_tol=float(max(0.0, args.inner_opt_accept_rel_tol)),
            inner_opt_guard_patience=int(max(1, args.inner_opt_guard_patience)),
            inner_opt_guard_check_interval=int(max(1, args.inner_opt_guard_check_interval)),
            inner_opt_alt_freeze_readout=bool(int(args.inner_opt_alt_freeze_readout)),
            inner_opt_grad_clip_norm=float(max(0.0, args.inner_opt_grad_clip_norm)),
            inner_opt_residual_clip_q=float(np.clip(args.inner_opt_residual_clip_q, 0.70, 0.999)),
            graph_cache=graph_cache,
        )

        outer_adapter_local, outer_meta_local = _build_outer_adapter(
            strategy=str(args.outer_strategy),
            pop_size=int(effective_pop_size),
            generations=int(max(1, generations_this_epoch)),
            portfolio_phases_csv=str(args.portfolio_phases),
            portfolio_weights_csv=str(args.portfolio_phase_weights),
            moead_neighborhood_size=int(max(2, args.moead_neighborhood_size)),
            moead_delta=float(args.moead_delta),
            moead_nr=int(max(1, args.moead_nr)),
            vns_k_max=int(max(1, args.vns_k_max)),
            vns_batch_size=int(effective_vns_batch_size),
        )

        rep_local = build_pipeline(problem_local)
        solver_local = ComposableSolver(
            problem_local,
            adapter=outer_adapter_local,
            representation_pipeline=rep_local,
        )
        if batched_eval_enabled:
            def _evaluate_population_batched(self: Any, population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                pop_arr = np.asarray(population, dtype=float)
                if pop_arr.ndim == 1:
                    pop_arr = pop_arr.reshape(1, -1)
                pop_size = int(pop_arr.shape[0])
                if bool(getattr(self, "snapshot_pre_evaluate_population", False)):
                    self._persist_snapshot(
                        population=pop_arr,
                        objectives=None,
                        violations=None,
                        include_pareto=True,
                        include_history=True,
                        include_decision_trace=True,
                        complete=False,
                    )
                objectives, violations = problem_local.evaluate_population_batch(pop_arr)
                self.evaluation_count += int(pop_size)
                self._persist_snapshot(
                    population=pop_arr,
                    objectives=objectives,
                    violations=violations,
                    include_pareto=True,
                    include_history=True,
                    include_decision_trace=True,
                    complete=True,
                )
                return objectives, violations

            solver_local.evaluate_population = types.MethodType(_evaluate_population_batched, solver_local)

        solver_local.max_steps = int(max(1, outer_meta_local.get("max_generations", generations_this_epoch)))
        solver_local.set_random_seed(int(seed_this_epoch))
        t0_local = time.perf_counter()
        run_local = solver_local.run()
        outer_sec_local = float(time.perf_counter() - t0_local)
        return problem_local, outer_meta_local, run_local, outer_sec_local

    epoch_generations: list[int] = [int(effective_generations)]
    if dynamic_pool_enabled and dynamic_pool_epochs > 1:
        base = int(effective_generations // dynamic_pool_epochs)
        rem = int(effective_generations - base * dynamic_pool_epochs)
        epoch_generations = [int(max(1, base + (1 if i < rem else 0))) for i in range(dynamic_pool_epochs)]

    outer_sec = 0.0
    outer_meta: dict[str, Any] = {"strategy": str(args.outer_strategy), "max_generations": int(effective_generations)}
    run: dict[str, Any] = {"status": "completed", "steps_executed": 0}
    top_cache: list[dict[str, Any]] = []
    problem: SymbolicSubsetSelectionProblem | None = None
    best_row: dict[str, Any] | None = None
    best_genome: list[dict[str, Any]] | None = None
    best_k = 0
    dynamic_epoch_logs: list[dict[str, Any]] = []

    for ep, gens_this in enumerate(epoch_generations):
        problem_ep, outer_meta_ep, run_ep, sec_ep = _run_outer_once(
            run_candidates=candidates,
            generations_this_epoch=int(gens_this),
            seed_this_epoch=int(args.seed + ep),
        )
        problem = problem_ep
        outer_sec += float(sec_ep)
        run = dict(run_ep)
        run["steps_executed"] = int(run.get("steps_executed", 0)) + int(sum(epoch_generations[:ep]))
        outer_meta = dict(outer_meta_ep)
        top_cache_ep = problem_ep.cache_top(topn=max(50, dynamic_top_cache_use))
        if not top_cache_ep:
            continue
        top_cache = list(top_cache_ep)

        row0 = dict(top_cache_ep[0])
        idx0 = [int(v) for v in row0.get("subset_idx", [])]
        genome0 = [{"name": candidates[i].name, "expr": dict(candidates[i].expr)} for i in idx0]
        if best_row is None:
            best_row = dict(row0)
            best_genome = list(genome0)
            best_k = int(row0.get("subset_size", len(idx0)))
        else:
            cur_key = (
                float(row0.get("obj_accuracy", float("inf"))),
                float(row0.get("obj_stability", float("inf"))),
                float(row0.get("obj_complexity", float("inf"))),
            )
            best_key = (
                float(best_row.get("obj_accuracy", float("inf"))),
                float(best_row.get("obj_stability", float("inf"))),
                float(best_row.get("obj_complexity", float("inf"))),
            )
            if cur_key < best_key:
                best_row = dict(row0)
                best_genome = list(genome0)
                best_k = int(row0.get("subset_size", len(idx0)))

        selected_keys: set[str] = set()
        for r in top_cache_ep[:dynamic_top_cache_use]:
            for j in [int(v) for v in r.get("subset_idx", [])]:
                if 0 <= int(j) < len(candidates):
                    selected_keys.add(json.dumps(candidates[int(j)].expr, sort_keys=True))

        n_new = 0
        n_after_prune = len(candidates)
        if dynamic_pool_enabled and ep < len(epoch_generations) - 1 and idx0:
            l2_ep = float(max(0.0, row0.get("tuned_l2", args.ridge_l2)))
            fit_ep = evaluate_genome_with_ridge(
                genome0,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_train,
                y_eval=y_train,
                l2=l2_ep,
            )
            pred_tr = _as_2d(np.asarray(fit_ep.get("pred_train"), dtype=float))
            res_tr = _as_2d(np.asarray(y_train - pred_tr, dtype=float))
            new_terms = _expand_candidate_pool_from_residual(
                X=X_train,
                y_residual=res_tr,
                feature_names=feature_names,
                base_genome=genome0,
                base_weight=_as_2d(np.asarray(fit_ep.get("weight"), dtype=float)),
                existing=candidates,
                max_new_terms=int(dynamic_expand_max_new),
                focus_top_features=int(dynamic_focus_top_features),
                partner_topk=int(dynamic_partner_topk),
            )
            n_new = int(len(new_terms))
            if n_new > 0:
                candidates = list(candidates) + list(new_terms)
            candidates = _prune_candidate_pool(
                candidates=candidates,
                keep_expr_keys=selected_keys,
                feature_names=feature_names,
                max_pool_size=int(dynamic_max_pool_size),
            )
            n_after_prune = int(len(candidates))

        dynamic_epoch_logs.append(
            {
                "epoch": int(ep + 1),
                "generations": int(gens_this),
                "duration_sec": float(sec_ep),
                "pool_size_before": int(len(problem_ep.candidates)),
                "pool_size_after": int(n_after_prune),
                "new_terms_added": int(n_new),
                "best_obj_accuracy": float(row0.get("obj_accuracy", float("inf"))),
                "best_subset_size": int(row0.get("subset_size", len(idx0))),
            }
        )

    if problem is None or best_row is None or best_genome is None:
        raise RuntimeError("outer search produced empty evaluation cache")

    best_decode_meta = {
        "rmse_mean": float(best_row.get("rmse_mean", float("inf"))),
        "rmse_std": float(best_row.get("rmse_std", float("inf"))),
        "obj_accuracy": float(best_row.get("obj_accuracy", float("inf"))),
        "obj_stability": float(best_row.get("obj_stability", float("inf"))),
        "obj_complexity": float(best_row.get("obj_complexity", float("inf"))),
        "decode_meta": _jsonable(best_row.get("decode_meta", {})),
        "tuned_l2": float(best_row.get("tuned_l2", max(0.0, args.ridge_l2))),
        "strict4_min_train_ratio": float(best_row.get("strict4_min_train_ratio", 0.08)),
    }
    best_subset_idx = [int(v) for v in best_row.get("subset_idx", [])]

    fit_final = _three_layer_fit_predict(
        genome=best_genome,
        X_train=X_train,
        y_train=y_train,
        X_eval=X_test,
        y_eval=y_test,
        l2=float(max(0.0, best_decode_meta["tuned_l2"])),
        inner_opt_enabled=bool(int(args.inner_opt_enabled)),
        inner_opt_adam_steps=int(max(0, args.inner_opt_adam_steps)),
        inner_opt_adam_lr=float(max(1e-8, args.inner_opt_adam_lr)),
        inner_opt_lbfgs_steps=int(max(0, args.inner_opt_lbfgs_steps)),
        inner_opt_lbfgs_lr=float(max(1e-8, args.inner_opt_lbfgs_lr)),
        inner_opt_accept_rmse_tol=float(max(0.0, args.inner_opt_accept_rmse_tol)),
        inner_opt_accept_rel_tol=float(max(0.0, args.inner_opt_accept_rel_tol)),
        inner_opt_guard_patience=int(max(1, args.inner_opt_guard_patience)),
        inner_opt_guard_check_interval=int(max(1, args.inner_opt_guard_check_interval)),
        inner_opt_alt_freeze_readout=bool(int(args.inner_opt_alt_freeze_readout)),
        inner_opt_grad_clip_norm=float(max(0.0, args.inner_opt_grad_clip_norm)),
        inner_opt_residual_clip_q=float(np.clip(args.inner_opt_residual_clip_q, 0.70, 0.999)),
    )
    sym_pred_train = _as_2d(np.asarray(fit_final.get("pred_train"), dtype=float))
    sym_pred_test = _as_2d(np.asarray(fit_final.get("pred_eval"), dtype=float))
    sym_rmse = float(_rmse(y_test, sym_pred_test))
    sym_mae = float(_mae(y_test, sym_pred_test))
    interval_alpha = float(np.clip(args.interval_alpha, 1e-6, 0.99))
    interval_method = str(args.interval_method)
    sym_interval_info: dict[str, Any] = {"method": str(interval_method)}
    if interval_method == "native_quantile_cqr":
        sym_lo, sym_hi, sym_interval_info = _build_native_quantile_interval(
            genome=best_genome,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_test,
            alpha=interval_alpha,
            calib_ratio=float(np.clip(args.interval_calib_ratio, 0.05, 0.4)),
            quantile_l2=float(max(0.0, args.interval_quantile_l2)),
        )
    else:
        sym_lo, sym_hi, q = _build_symmetric_interval(
            y_train=y_train,
            pred_train=sym_pred_train,
            pred_eval=sym_pred_test,
            alpha=interval_alpha,
        )
        sym_interval_info = {"method": "symmetric_residual", "conformal_qhat": float(q)}
    sym_interval = _interval_metrics(
        y_true=y_test,
        lower=sym_lo,
        upper=sym_hi,
        alpha=interval_alpha,
    )

    xgb = XGBoostSurrogateTrainer(
        config=XGBoostTrainerConfig(
            artifact_id="subset_bridge_xgb_baseline",
            n_estimators=360,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_seed=42,
        )
    )
    xgb_art = xgb.fit(
        ProcessedDataset(
            X_train=X_train,
            y_train=y_train,
            feature_names=feature_names,
            target_names=(str(args.target_col),),
        )
    )
    xgb_pred = np.asarray(xgb_art.predict(X_test), dtype=float).reshape(-1, 1)
    xgb_rmse = _rmse(y_test, xgb_pred)
    xgb_mae = _mae(y_test, xgb_pred)
    xgb_train_pred = np.asarray(xgb_art.predict(X_train), dtype=float).reshape(-1, 1)
    xgb_lo, xgb_hi, xgb_calib_q = _build_symmetric_interval(
        y_train=y_train,
        pred_train=xgb_train_pred,
        pred_eval=xgb_pred,
        alpha=interval_alpha,
    )
    xgb_interval = _interval_metrics(
        y_true=y_test,
        lower=xgb_lo,
        upper=xgb_hi,
        alpha=interval_alpha,
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "config": {
            "csv_path": str(args.csv_path),
            "target_col": str(args.target_col),
            "test_fold_col": str(args.test_fold_col),
            "pop_size": int(args.pop_size),
            "generations": int(args.generations),
            "effective_pop_size": int(effective_pop_size),
            "effective_generations": int(effective_generations),
            "rolling_folds": int(args.rolling_folds),
            "rolling_val_ratio": float(args.rolling_val_ratio),
            "max_terms": int(args.max_terms),
            "ridge_l2": float(args.ridge_l2),
            "strict4_branch_mode_requested": bool(args.strict4_branch_mode),
            "strict4_branch_mode_enabled": bool(strict4_enabled),
            "strict4_min_branch_train": int(args.strict4_min_branch_train),
            "strict4_branch_parallel_workers": int(args.strict4_branch_parallel_workers),
            "effective_strict4_branch_parallel_workers": int(effective_strict4_workers),
            "seed": int(args.seed),
            "outer_strategy": str(args.outer_strategy),
            "portfolio_phases": str(args.portfolio_phases),
            "portfolio_phase_weights": str(args.portfolio_phase_weights),
            "moead_neighborhood_size": int(args.moead_neighborhood_size),
            "moead_delta": float(args.moead_delta),
            "moead_nr": int(args.moead_nr),
            "vns_k_max": int(args.vns_k_max),
            "vns_batch_size": int(args.vns_batch_size),
            "effective_vns_batch_size": int(effective_vns_batch_size),
            "batched_eval_enabled": bool(batched_eval_enabled),
            "reinvest_search": bool(reinvest_enabled),
            "reinvest_pop_mult": float(args.reinvest_pop_mult),
            "reinvest_gen_mult": float(args.reinvest_gen_mult),
            "reinvest_strict4_workers_mult": float(args.reinvest_strict4_workers_mult),
            "dynamic_pool_enabled": bool(dynamic_pool_enabled),
            "dynamic_pool_epochs": int(dynamic_pool_epochs),
            "dynamic_init_minimal": bool(dynamic_init_minimal),
            "dynamic_expand_max_new": int(dynamic_expand_max_new),
            "dynamic_focus_top_features": int(dynamic_focus_top_features),
            "dynamic_partner_topk": int(dynamic_partner_topk),
            "dynamic_top_cache_use": int(dynamic_top_cache_use),
            "dynamic_max_pool_size": int(dynamic_max_pool_size),
            "graph_cache_enabled": bool(graph_cache_enabled),
            "graph_cache_backend": str(graph_cache_backend),
            "graph_cache_db_path": str(graph_cache_db_path),
            "graph_cache_namespace": str(args.graph_cache_namespace),
            "graph_cache_persist_values": bool(int(args.graph_cache_persist_values)),
            "interval_alpha": float(interval_alpha),
            "interval_method": str(interval_method),
            "interval_calib_ratio": float(args.interval_calib_ratio),
            "interval_quantile_l2": float(args.interval_quantile_l2),
            "safe_log1p_abs": bool(safe_log1p_abs_enabled),
            "safe_exp_clip": bool(safe_exp_clip_enabled),
            "safe_reciprocal": bool(safe_reciprocal_enabled),
            "safe_exp_clip_k": float(safe_exp_clip_k),
            "safe_reciprocal_eps": float(safe_reciprocal_eps),
            "lag_feature_enabled": bool(lag_enabled),
            "lag_orders": [int(v) for v in lag_orders],
            "lag_sources": sorted([str(v) for v in lag_source_set]),
            "lag_added_features": list(lag_added_features),
            "lag_cross_enabled": bool(lag_cross_enabled),
            "lag_cross_quantiles": [float(v) for v in lag_cross_q],
            "lag_cross_added_features": list(lag_cross_added_features),
            "drop_same_day_flow_speed_occ": bool(int(args.drop_same_day_flow_speed_occ)),
            "drop_feature_list": [s.strip() for s in str(args.drop_feature_list).split(",") if s.strip()],
            "dropped_features": list(dropped_features),
            "inner_opt_enabled": bool(int(args.inner_opt_enabled)),
            "inner_opt_adam_steps": int(args.inner_opt_adam_steps),
            "inner_opt_adam_lr": float(args.inner_opt_adam_lr),
            "inner_opt_lbfgs_steps": int(args.inner_opt_lbfgs_steps),
            "inner_opt_lbfgs_lr": float(args.inner_opt_lbfgs_lr),
            "inner_opt_accept_rmse_tol": float(args.inner_opt_accept_rmse_tol),
            "inner_opt_accept_rel_tol": float(args.inner_opt_accept_rel_tol),
            "inner_opt_guard_patience": int(args.inner_opt_guard_patience),
            "inner_opt_guard_check_interval": int(args.inner_opt_guard_check_interval),
            "inner_opt_alt_freeze_readout": bool(int(args.inner_opt_alt_freeze_readout)),
            "inner_opt_grad_clip_norm": float(args.inner_opt_grad_clip_norm),
            "inner_opt_residual_clip_q": float(args.inner_opt_residual_clip_q),
            "outer_decision_encoding": "expanded_structure_plus_hyperparams_v1",
            "nesting_architecture": "outer_structure_search -> middle_param_optimizer -> inner_rolling_eval",
        },
        "dataset": {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_features_raw": int(n_features_raw),
            "feature_names_raw": list(feature_names_raw),
            "feature_names": list(feature_names),
            "dropped_features": list(dropped_features),
            "lag_added_features": list(lag_added_features),
            "lag_cross_added_features": list(lag_cross_added_features),
        },
        "outer_search": {
            "duration_sec": float(outer_sec),
            "outer_meta": _jsonable({**dict(outer_meta), "dynamic_pool_epochs": dynamic_epoch_logs}),
            "n_candidates": int(len(candidates)),
            "n_families": int(len(problem.families)),
            "families": [str(v) for v in problem.families],
            "run_result": _jsonable(run),
            "n_cached_evals": int(len(problem._cache)),
            "top_cache": top_cache[:20],
            "graph_cache": _jsonable(graph_cache.snapshot()),
        },
        "best_solution": {
            "k_decoded": int(best_k),
            "subset_size": int(len(best_subset_idx)),
            "subset_idx": [int(i) for i in best_subset_idx],
            "subset_names": _jsonable(best_row.get("subset_names", [])),
            "subset_families": _jsonable(best_row.get("subset_families", [])),
            "decode_meta": _jsonable(best_decode_meta),
            "obj_accuracy": float(best_row.get("obj_accuracy", float("inf"))),
            "obj_stability": float(best_row.get("obj_stability", float("inf"))),
            "obj_complexity": float(best_row.get("obj_complexity", float("inf"))),
            "inner_opt_info": _jsonable(fit_final.get("inner_opt_info", {})),
        },
        "test_compare": {
            "symbolic_subset_rmse": float(sym_rmse),
            "symbolic_subset_mae": float(sym_mae),
            "xgboost_rmse": float(xgb_rmse),
            "xgboost_mae": float(xgb_mae),
            "delta_symbolic_minus_xgb": float(sym_rmse - xgb_rmse),
            "interval_metrics": {
                "symbolic": {
                    **_jsonable(sym_interval_info),
                    **_jsonable(sym_interval),
                },
                "xgboost": {
                    "calib_abs_residual_q": float(xgb_calib_q),
                    **_jsonable(xgb_interval),
                },
            },
        },
    }
    report_path = out_root / "summary.json"
    report_path.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    graph_cache.close()

    print("NSGABLACK_SYMBOLIC_SUBSET_BRIDGE_DONE")
    print(f"summary={report_path}")
    print(
        "rmse: "
        f"symbolic_subset={float(sym_rmse):.6f}, "
        f"xgboost={float(xgb_rmse):.6f}, "
        f"delta={float(sym_rmse - xgb_rmse):.6f}"
    )
    print(
        "interval: "
        f"alpha={float(interval_alpha):.3f} | "
        f"symbolic(PICP={float(sym_interval['picp']):.4f}, PINAW={float(sym_interval['pinaw']):.4f}, IS={float(sym_interval['interval_score']):.4f}) | "
        f"xgb(PICP={float(xgb_interval['picp']):.4f}, PINAW={float(xgb_interval['pinaw']):.4f}, IS={float(xgb_interval['interval_score']):.4f})"
    )

__all__ = ['main']

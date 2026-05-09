from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DESKTOP = Path(__file__).resolve().parents[3]
NSGABLACK_ROOT = DESKTOP / "nsgablack"
MLBLACK_ROOT = DESKTOP / "mlblack"
if str(NSGABLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(NSGABLACK_ROOT))
if str(MLBLACK_ROOT) not in sys.path:
    sys.path.insert(0, str(MLBLACK_ROOT))

from bias import (
    BranchPolicyConfig,
    DynamicPoolPolicyConfig,
    ObjectivePolicyConfig,
    build_epoch_generations,
    build_interval_row_objective_key,
    collect_selected_expr_keys,
    resolve_branch_workers,
    should_expand_dynamic_pool,
)
from conditional import (
    AutoThresholdBindingSpec,
    BinaryGateBinding,
    ConditionalComposerConfig,
    ConditionalConfig,
    RoutePlusPrimitivesSpec,
    RouteThenFormulaSpec,
    ThresholdPrimitiveBinding,
    build_auto_threshold_bindings,
    build_feature_role_config,
)
from conditional.router import resolve_router
from core.symbolic.feature_space.regime_router import (
    Strict4RouterSpec,
    map_to_strict4_regime,
    resolve_regime,
)
from nowcasting_work_ci.mlblack_side.problem.domain_router import (
    WORK_CI_STRICT4_POLICY,
    build_work_ci_conditional_router_policy,
)
from nowcasting_work_ci.mlblack_side.runtime.assembly import reg_conditional
from pipeline.feature_space import build_conditional_candidate_terms
from pipeline.feature_space import _expand_candidate_pool_from_residual


class _Candidate:
    def __init__(self, expr: dict[str, object]) -> None:
        self.expr = dict(expr)


def test_objective_policy_prefers_threshold_satisfied_row() -> None:
    cfg = ObjectivePolicyConfig(coverage_error_threshold=0.05)
    threshold_hit = {
        "obj_coverage_error": 0.04,
        "obj_pinaw": 0.30,
        "obj_interval_score": 30.0,
    }
    threshold_miss = {
        "obj_coverage_error": 0.08,
        "obj_pinaw": 0.10,
        "obj_interval_score": 10.0,
    }
    assert build_interval_row_objective_key(threshold_hit, cfg=cfg) < build_interval_row_objective_key(
        threshold_miss,
        cfg=cfg,
    )


def test_build_epoch_generations_evenly_splits_budget() -> None:
    cfg = DynamicPoolPolicyConfig(enabled=True, epochs=4)
    assert build_epoch_generations(10, cfg=cfg) == [3, 3, 2, 2]


def test_collect_selected_expr_keys_uses_top_cache_limit() -> None:
    candidates = [
        _Candidate({"name": "a"}),
        _Candidate({"name": "b"}),
        _Candidate({"name": "c"}),
    ]
    rows = [
        {"subset_idx": [0, 1]},
        {"subset_idx": [2]},
    ]
    keys = collect_selected_expr_keys(rows, candidates, top_cache_use=1)
    assert len(keys) == 2
    assert '{"name": "a"}' in keys
    assert '{"name": "b"}' in keys


def test_should_expand_dynamic_pool_only_before_last_epoch() -> None:
    cfg = DynamicPoolPolicyConfig(enabled=True, epochs=3)
    epochs = build_epoch_generations(9, cfg=cfg)
    assert should_expand_dynamic_pool(cfg, epoch_idx=0, epoch_generations=epochs, has_active_subset=True)
    assert not should_expand_dynamic_pool(cfg, epoch_idx=2, epoch_generations=epochs, has_active_subset=True)


def test_resolve_branch_workers_respects_reinvest_policy() -> None:
    cfg = BranchPolicyConfig(enabled=True, parallel_workers=4)
    assert resolve_branch_workers(cfg, reinvest_enabled=False, reinvest_mult=2.0) == 4
    assert resolve_branch_workers(cfg, reinvest_enabled=True, reinvest_mult=1.5) == 6


def test_core_default_regime_policy_uses_generic_fallback() -> None:
    generic = Strict4RouterSpec()
    assert map_to_strict4_regime((1, 0, 0, 0), router_spec=generic) == (0, 0, 0, 0)


def test_work_ci_policy_keeps_traffic_holiday_canonicalization() -> None:
    assert map_to_strict4_regime((1, 1, 1, 0), router_spec=WORK_CI_STRICT4_POLICY) == (1, 1, 0, 0)
    assert map_to_strict4_regime((1, 0, 1, 1), router_spec=WORK_CI_STRICT4_POLICY) == (1, 0, 1, 0)
    assert map_to_strict4_regime((0, 0, 0, 1), router_spec=WORK_CI_STRICT4_POLICY) == (0, 0, 0, 1)


def test_work_ci_policy_can_be_adapted_into_conditional_router() -> None:
    policy = build_work_ci_conditional_router_policy()
    assert policy.gate_names == WORK_CI_STRICT4_POLICY.gate_names
    assert policy.route_order == WORK_CI_STRICT4_POLICY.regime_order
    assert policy.canonicalize((1, 1, 1, 0)) == (1, 1, 0, 0)


def test_conditional_router_resolution_uses_policy_gate_names() -> None:
    policy = build_work_ci_conditional_router_policy()
    resolution = resolve_router(policy.gate_names, enabled=True, policy=policy)
    assert resolution.enabled
    assert resolution.gate_idx == (0, 1, 2, 3)


def test_core_regime_resolution_reuses_conditional_router_contract() -> None:
    resolution = resolve_regime(
        WORK_CI_STRICT4_POLICY.gate_names,
        enabled=True,
        regime_policy=WORK_CI_STRICT4_POLICY,
    )
    assert resolution.enabled
    assert resolution.gate_idx == (0, 1, 2, 3)


def test_conditional_config_builds_route_plus_primitives_spec() -> None:
    policy = build_work_ci_conditional_router_policy()
    feature_names = (*policy.gate_names, "ci_lag1", "avg_speed_lag1")
    roles = build_feature_role_config(
        feature_names,
        router_features=policy.gate_names,
        threshold_features=("ci_lag1",),
    )
    cfg = ConditionalConfig(
        enabled=True,
        feature_roles=roles,
        router_policy=policy,
        binary_gates=(BinaryGateBinding(feature_name=policy.gate_names[0]),),
        threshold_primitives=(
            ThresholdPrimitiveBinding(
                feature_name="ci_lag1",
                cuts=(0.5,),
                primitive_family="hinge",
                multiplier_feature="avg_speed_lag1",
            ),
        ),
        composer=ConditionalComposerConfig(mode="route_plus_primitives"),
    )
    primitive_specs = cfg.primitive_specs()
    assert len(primitive_specs) == 2
    assert primitive_specs[0].family == "gate_binary"
    assert primitive_specs[1].family == "piecewise_hinge"
    composer_spec = cfg.composer_spec()
    assert isinstance(composer_spec, RoutePlusPrimitivesSpec)
    assert composer_spec.router_policy.gate_names == policy.gate_names


def test_runtime_reg_conditional_exposes_default_feature_roles() -> None:
    feature_names = (
        *WORK_CI_STRICT4_POLICY.gate_names,
        "ci_lag1",
        "avg_speed_lag1",
        "total_flow_lag2",
    )
    runtime = reg_conditional(
        SimpleNamespace(),
        feature_names=feature_names,
        branch_runtime={"strict4_enabled": True},
    )
    role_index = runtime["role_index"]
    assert "router" in role_index["is_holiday_day_or_window"]
    assert "threshold" in role_index["ci_lag1"]
    assert "smooth" in role_index["ci_lag1"]
    assert len(runtime["config"].primitive_specs()) == len(WORK_CI_STRICT4_POLICY.gate_names)
    assert isinstance(runtime["composer_spec"], RouteThenFormulaSpec)


def test_build_auto_threshold_bindings_generates_quantile_hinges() -> None:
    feature_names = ("ci_lag1", "avg_speed_lag1", "shock_ci", "volatility_ci")
    X = np.asarray(
        [
            [1.0, 31.0, 0.10, 0.20],
            [2.0, 30.0, 0.20, 0.25],
            [3.0, 29.0, 0.60, 0.30],
            [4.0, 28.0, 0.80, 0.40],
            [5.0, 27.0, 1.20, 0.55],
            [6.0, 26.0, 1.60, 0.70],
        ],
        dtype=float,
    )
    bindings = build_auto_threshold_bindings(
        X,
        feature_names,
        threshold_features=("ci_lag1", "shock_ci"),
        per_feature_specs={
            "ci_lag1": AutoThresholdBindingSpec(
                primitive_family="hinge",
                quantiles=(0.33, 0.66),
                directions=("positive", "negative"),
                multiplier_feature="avg_speed_lag1",
            ),
            "shock_ci": AutoThresholdBindingSpec(
                primitive_family="hinge",
                quantiles=(0.8,),
                directions=("positive",),
            ),
        },
    )
    assert len(bindings) == 3
    assert bindings[0].multiplier_feature == "avg_speed_lag1"
    assert bindings[0].direction == "positive"
    assert bindings[1].direction == "negative"
    assert bindings[2].feature_name == "shock_ci"


def test_runtime_reg_conditional_with_training_matrix_adds_auto_threshold_primitives() -> None:
    feature_names = (
        *WORK_CI_STRICT4_POLICY.gate_names,
        "ci_lag1",
        "avg_speed_lag1",
        "shock_ci",
        "volatility_ci",
    )
    X = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 30.0, 0.10, 0.20],
            [1.0, 0.0, 0.0, 0.0, 2.0, 29.0, 0.30, 0.25],
            [0.0, 1.0, 0.0, 0.0, 3.0, 28.0, 0.60, 0.35],
            [0.0, 0.0, 1.0, 0.0, 4.0, 27.0, 0.90, 0.55],
            [0.0, 0.0, 0.0, 1.0, 5.0, 26.0, 1.20, 0.80],
            [1.0, 0.0, 0.0, 0.0, 6.0, 25.0, 1.50, 1.10],
        ],
        dtype=float,
    )
    runtime = reg_conditional(
        SimpleNamespace(),
        feature_names=feature_names,
        branch_runtime={"strict4_enabled": True},
        X_train=X,
    )
    specs = runtime["config"].primitive_specs()
    families = [spec.family for spec in specs]
    assert families.count("gate_binary") == len(WORK_CI_STRICT4_POLICY.gate_names)
    assert "piecewise_hinge" in families
    assert len(specs) > len(WORK_CI_STRICT4_POLICY.gate_names)


def test_conditional_candidate_terms_lower_gate_and_hinge_specs() -> None:
    policy = build_work_ci_conditional_router_policy()
    feature_names = (*policy.gate_names, "ci_lag1", "avg_speed_lag1")
    roles = build_feature_role_config(
        feature_names,
        router_features=policy.gate_names,
        threshold_features=("ci_lag1",),
    )
    cfg = ConditionalConfig(
        enabled=True,
        feature_roles=roles,
        router_policy=policy,
        binary_gates=(BinaryGateBinding(feature_name=policy.gate_names[0]),),
        threshold_primitives=(
            ThresholdPrimitiveBinding(
                feature_name="ci_lag1",
                cuts=(1.5,),
                primitive_family="hinge",
                multiplier_feature="avg_speed_lag1",
            ),
        ),
    )
    X = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 30.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 28.0],
            [0.0, 1.0, 0.0, 0.0, 3.0, 26.0],
        ],
        dtype=float,
    )
    y = np.asarray([10.0, 12.0, 11.0], dtype=float)
    terms = build_conditional_candidate_terms(
        X,
        y,
        feature_names=feature_names,
        conditional_config=cfg,
    )
    families = {term.family for term in terms}
    assert "gate_binary" in families
    assert "piecewise_hinge" in families
    hinge_terms = [term for term in terms if term.family == "piecewise_hinge"]
    assert hinge_terms
    assert len(hinge_terms[0].features) == 2


def test_dynamic_pool_expansion_reuses_conditional_auto_cut_family() -> None:
    policy = build_work_ci_conditional_router_policy()
    feature_names = (*policy.gate_names, "ci_lag1", "avg_speed_lag1")
    roles = build_feature_role_config(
        feature_names,
        router_features=policy.gate_names,
        threshold_features=("ci_lag1",),
    )
    cfg = ConditionalConfig(
        enabled=True,
        feature_roles=roles,
        router_policy=policy,
        threshold_primitives=(
            ThresholdPrimitiveBinding(
                feature_name="ci_lag1",
                cuts=(3.5,),
                primitive_family="hinge",
                direction="positive",
                multiplier_feature="avg_speed_lag1",
            ),
        ),
    )
    X = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 30.0],
            [1.0, 0.0, 0.0, 0.0, 2.0, 29.0],
            [0.0, 1.0, 0.0, 0.0, 3.0, 28.0],
            [0.0, 0.0, 1.0, 0.0, 4.0, 27.0],
            [0.0, 0.0, 0.0, 1.0, 5.0, 26.0],
            [1.0, 0.0, 0.0, 0.0, 6.0, 25.0],
            [0.0, 1.0, 0.0, 0.0, 7.0, 24.0],
            [0.0, 0.0, 1.0, 0.0, 8.0, 23.0],
        ],
        dtype=float,
    )
    residual = np.asarray([[0.0], [0.1], [0.2], [0.4], [1.0], [1.8], [2.8], [4.0]], dtype=float)
    base_genome = [{"name": "x4", "expr": {"type": "feature", "index": 4}}]
    base_weight = np.asarray([[1.0]], dtype=float)
    existing = []
    new_terms = _expand_candidate_pool_from_residual(
        X=X,
        y_residual=residual,
        feature_names=feature_names,
        base_genome=base_genome,
        base_weight=base_weight,
        existing=existing,
        max_new_terms=24,
        focus_top_features=3,
        partner_topk=2,
        conditional_config=cfg,
    )
    guided_hinges = [term for term in new_terms if term.family == "piecewise_hinge" and len(term.features) == 2]
    assert guided_hinges

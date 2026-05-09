from __future__ import annotations

from my_project.known_relation_symbolic.problem.generators import (
    _build_arrhenius_gate_like,
    _build_coupled_reaction_transport_like,
    _build_ideal_gas_like,
    _build_ohm_like,
    _build_periodic_gate_like,
    _build_redundant_proxy_control,
)
from my_project.known_relation_symbolic.problem.specs import KnownRelationBenchmarkDefinition


KNOWN_RELATION_BENCHMARKS: dict[str, KnownRelationBenchmarkDefinition] = {
    "ohm_like": KnownRelationBenchmarkDefinition(
        key="ohm_like",
        description="Ratio + periodic + piecewise + material bias benchmark mirroring an ohm-like decomposition.",
        feature_names=("voltage", "resistance", "temperature", "material_bias", "sensor_noise"),
        target_names=("current",),
        truth_expression=(
            "current = 1.42 * voltage/safe(resistance)"
            " + 0.58 * sin(temperature)"
            " + 0.72 * relu(temperature-0.25)"
            " - 0.44 * relu(-temperature-0.35)"
            " - 0.18 * material_bias + noise"
        ),
        strict_contract=(
            "safe_ratio(voltage,resistance)",
            "sin(temperature)",
            "piecewise_hinge(temperature)",
            "material_bias",
        ),
        phase_equivalent_contract=(
            "safe_ratio(voltage,resistance)",
            "periodic_phase_equivalent(temperature)",
            "piecewise_hinge(temperature)",
            "material_bias",
        ),
        family_level_contract=(
            "ratio_family(voltage,resistance)",
            "periodic_family(temperature)",
            "piecewise_gate_family(temperature)",
            "linear_feature_family(material_bias)",
        ),
        gate_feature_names=("temperature",),
        periodic_feature_names=("temperature",),
        enable_piecewise_basis=True,
        builder=_build_ohm_like,
    ),
    "ideal_gas_like": KnownRelationBenchmarkDefinition(
        key="ideal_gas_like",
        description="Product-over-volume benchmark for orthogonal interaction and ratio discovery.",
        feature_names=("amount", "temperature", "volume", "material_bias", "sensor_noise"),
        target_names=("pressure",),
        truth_expression=(
            "pressure = 2.15 * (amount * temperature)/safe(volume)"
            " - 0.22 * material_bias + noise"
        ),
        strict_contract=(
            "product_ratio(amount,temperature,volume)",
            "material_bias",
        ),
        phase_equivalent_contract=(
            "product_ratio(amount,temperature,volume)",
            "material_bias",
        ),
        family_level_contract=(
            "product_ratio_family(amount,temperature,volume)",
            "linear_feature_family(material_bias)",
        ),
        gate_feature_names=tuple(),
        periodic_feature_names=tuple(),
        enable_piecewise_basis=False,
        builder=_build_ideal_gas_like,
    ),
    "arrhenius_gate_like": KnownRelationBenchmarkDefinition(
        key="arrhenius_gate_like",
        description="Arrhenius-style exp-over-ratio plus warm-regime gate and catalyst bias.",
        feature_names=("activation_energy", "temperature", "catalyst_bias", "pressure_bias", "sensor_noise"),
        target_names=("rate",),
        truth_expression=(
            "rate = 1.75 * exp(-activation_energy/safe(temperature))"
            " + 0.63 * relu(temperature-1.8)"
            " - 0.27 * catalyst_bias + noise"
        ),
        strict_contract=(
            "exp_ratio(activation_energy,temperature)",
            "piecewise_hinge(temperature)",
            "catalyst_bias",
        ),
        phase_equivalent_contract=(
            "exp_ratio(activation_energy,temperature)",
            "piecewise_hinge(temperature)",
            "catalyst_bias",
        ),
        family_level_contract=(
            "exp_ratio_family(activation_energy,temperature)",
            "piecewise_gate_family(temperature)",
            "linear_feature_family(catalyst_bias)",
        ),
        gate_feature_names=("temperature",),
        periodic_feature_names=tuple(),
        enable_piecewise_basis=True,
        builder=_build_arrhenius_gate_like,
    ),
    "periodic_gate_like": KnownRelationBenchmarkDefinition(
        key="periodic_gate_like",
        description="Periodic signal plus phase gate and material-bias nuisance for phase-equivalent recovery.",
        feature_names=("phase_angle", "load", "material_bias", "sensor_noise", "trend_bias"),
        target_names=("signal",),
        truth_expression=(
            "signal = 0.94 * sin(phase_angle)"
            " + 0.58 * relu(phase_angle-0.45)"
            " - 0.21 * material_bias + noise"
        ),
        strict_contract=(
            "sin(phase_angle)",
            "piecewise_hinge(phase_angle)",
            "material_bias",
        ),
        phase_equivalent_contract=(
            "periodic_phase_equivalent(phase_angle)",
            "piecewise_hinge(phase_angle)",
            "material_bias",
        ),
        family_level_contract=(
            "periodic_family(phase_angle)",
            "piecewise_gate_family(phase_angle)",
            "linear_feature_family(material_bias)",
        ),
        gate_feature_names=("phase_angle",),
        periodic_feature_names=("phase_angle",),
        enable_piecewise_basis=True,
        builder=_build_periodic_gate_like,
    ),
    "redundant_proxy_control": KnownRelationBenchmarkDefinition(
        key="redundant_proxy_control",
        description="Primary-signal benchmark with a redundant proxy feature to test semantic deduplication.",
        feature_names=("primary_signal", "primary_signal_proxy", "phase_angle", "drift_bias", "sensor_noise"),
        target_names=("response",),
        truth_expression=(
            "response = 1.26 * primary_signal"
            " + 0.61 * sin(phase_angle)"
            " + 0.44 * relu(primary_signal-0.1)"
            " - 0.17 * drift_bias + noise"
        ),
        strict_contract=(
            "primary_signal",
            "sin(phase_angle)",
            "piecewise_hinge(primary_signal)",
            "drift_bias",
        ),
        phase_equivalent_contract=(
            "primary_signal",
            "periodic_phase_equivalent(phase_angle)",
            "piecewise_hinge(primary_signal)",
            "drift_bias",
        ),
        family_level_contract=(
            "linear_feature_family(primary_signal)",
            "periodic_family(phase_angle)",
            "piecewise_gate_family(primary_signal)",
            "linear_feature_family(drift_bias)",
        ),
        gate_feature_names=("primary_signal",),
        periodic_feature_names=("phase_angle",),
        enable_piecewise_basis=True,
        builder=_build_redundant_proxy_control,
    ),
    "coupled_reaction_transport_like": KnownRelationBenchmarkDefinition(
        key="coupled_reaction_transport_like",
        description=(
            "Composite known-relation benchmark combining a product-ratio transport trunk, "
            "Arrhenius realization, periodic phase channel, load gate branch, and proxy distractor."
        ),
        feature_names=(
            "flow_rate",
            "concentration",
            "temperature",
            "activation_energy",
            "phase_angle",
            "load",
            "load_proxy",
            "catalyst_bias",
            "sensor_noise",
        ),
        target_names=("reaction_output",),
        truth_expression=(
            "reaction_output = 0.86 * (flow_rate * concentration)/safe(temperature)"
            " + 1.34 * exp(-activation_energy/safe(temperature))"
            " + 0.52 * sin(phase_angle)"
            " + 0.43 * relu(load-0.28)"
            " - 0.22 * catalyst_bias + noise"
        ),
        strict_contract=(
            "product_ratio(flow_rate,concentration,temperature)",
            "exp_ratio(activation_energy,temperature)",
            "sin(phase_angle)",
            "piecewise_hinge(load)",
            "catalyst_bias",
        ),
        phase_equivalent_contract=(
            "product_ratio(flow_rate,concentration,temperature)",
            "exp_ratio(activation_energy,temperature)",
            "periodic_phase_equivalent(phase_angle)",
            "piecewise_hinge(load)",
            "catalyst_bias",
        ),
        family_level_contract=(
            "product_ratio_family(flow_rate,concentration,temperature)",
            "exp_ratio_family(activation_energy,temperature)",
            "periodic_family(phase_angle)",
            "piecewise_gate_family(load)",
            "linear_feature_family(catalyst_bias)",
        ),
        gate_feature_names=("load",),
        periodic_feature_names=("phase_angle",),
        enable_piecewise_basis=True,
        builder=_build_coupled_reaction_transport_like,
    ),
}




def known_relation_benchmark_keys() -> tuple[str, ...]:
    return tuple(KNOWN_RELATION_BENCHMARKS.keys())


def get_known_relation_benchmark(key: str) -> KnownRelationBenchmarkDefinition:
    normalized = str(key or "").strip().lower()
    if normalized not in KNOWN_RELATION_BENCHMARKS:
        choices = ", ".join(known_relation_benchmark_keys())
        raise KeyError(f"Unknown known-relation benchmark '{key}'. Expected one of: {choices}")
    return KNOWN_RELATION_BENCHMARKS[normalized]


__all__ = [
    "KNOWN_RELATION_BENCHMARKS",
    "get_known_relation_benchmark",
    "known_relation_benchmark_keys",
]

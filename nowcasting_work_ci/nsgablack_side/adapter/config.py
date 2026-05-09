from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from nsgablack.adapters import (
    MOEADAdapter,
    MOEADConfig,
    NSGA2Adapter,
    NSGA2Config,
    SerialPhaseSpec,
    SerialStrategyConfig,
    StrategyChainAdapter,
    VNSAdapter,
    VNSConfig,
)


def _parse_csv_list(raw: str) -> list[str]:
    return [v.strip() for v in str(raw).split(",") if v.strip()]


def _parse_csv_floats(raw: str) -> list[float]:
    out: list[float] = []
    for s in str(raw).split(","):
        ss = s.strip()
        if not ss:
            continue
        try:
            out.append(float(ss))
        except Exception:
            continue
    return out


def _allocate_phase_steps(total_steps: int, n_phases: int, weights: Sequence[float]) -> list[int]:
    total = int(max(1, total_steps))
    n = int(max(1, n_phases))
    if len(weights) < n:
        weights = list(weights) + [1.0] * int(n - len(weights))
    w = np.asarray([max(1e-8, float(weights[i])) for i in range(n)], dtype=float)
    w = w / float(np.sum(w))

    steps = [max(1, int(round(total * float(wi)))) for wi in w]
    while sum(steps) > total and any(v > 1 for v in steps):
        idx = int(np.argmax(np.asarray(steps, dtype=int)))
        if steps[idx] > 1:
            steps[idx] -= 1
        else:
            break
    while sum(steps) < total:
        idx = int(np.argmin(np.asarray(steps, dtype=int)))
        steps[idx] += 1
    return [int(v) for v in steps]


def build_outer_adapter(
    *,
    strategy: str,
    pop_size: int,
    generations: int,
    seed: int | None,
    portfolio_phases_csv: str,
    portfolio_weights_csv: str,
    moead_neighborhood_size: int,
    moead_delta: float,
    moead_nr: int,
    vns_k_max: int,
    vns_batch_size: int,
) -> tuple[Any, dict[str, Any]]:
    mode = str(strategy).strip().lower()
    pop = int(max(4, pop_size))
    gens = int(max(1, generations))

    if mode == "nsga2":
        adapter = NSGA2Adapter(
            config=NSGA2Config(
                population_size=int(pop),
                offspring_size=int(pop),
                crossover_rate=0.90,
                objective_aggregation="sum",
            )
        )
        return adapter, {"strategy": "nsga2", "max_generations": int(gens)}

    if mode == "moead":
        adapter = MOEADAdapter(
            config=MOEADConfig(
                population_size=int(pop),
                neighborhood_size=int(max(2, moead_neighborhood_size)),
                batch_size=int(pop),
                delta=float(np.clip(moead_delta, 0.05, 1.0)),
                nr=int(max(1, moead_nr)),
                decomposition="tchebycheff",
                random_seed=None if seed is None else int(seed),
            )
        )
        return adapter, {"strategy": "moead", "max_generations": int(gens)}

    if mode == "vns":
        adapter = VNSAdapter(
            config=VNSConfig(
                batch_size=int(max(4, vns_batch_size)),
                k_max=int(max(1, vns_k_max)),
                base_sigma=0.15,
                scale=1.45,
                max_sigma=2.0,
                objective_aggregation="sum",
            )
        )
        return adapter, {"strategy": "vns", "max_generations": int(gens)}

    phase_names = [v.lower() for v in _parse_csv_list(portfolio_phases_csv)]
    if not phase_names:
        phase_names = ["nsga2", "moead", "vns"]
    phase_weights = _parse_csv_floats(portfolio_weights_csv)
    if not phase_weights:
        phase_weights = [2.0, 1.0, 1.0]
    phase_steps = _allocate_phase_steps(gens, len(phase_names), phase_weights)

    phases: list[SerialPhaseSpec] = []
    phase_meta: list[dict[str, Any]] = []
    for i, (nm, st) in enumerate(zip(phase_names, phase_steps)):
        name = str(nm).strip().lower()
        if name == "nsga2":
            adapter_i = NSGA2Adapter(
                config=NSGA2Config(
                    population_size=int(pop),
                    offspring_size=int(pop),
                    crossover_rate=0.90,
                    objective_aggregation="sum",
                ),
                name=f"nsga2_phase_{i}",
            )
        elif name == "moead":
            adapter_i = MOEADAdapter(
                config=MOEADConfig(
                    population_size=int(pop),
                    neighborhood_size=int(max(2, moead_neighborhood_size)),
                    batch_size=int(pop),
                    delta=float(np.clip(moead_delta, 0.05, 1.0)),
                    nr=int(max(1, moead_nr)),
                    decomposition="tchebycheff",
                    random_seed=None if seed is None else int(seed + i),
                ),
                name=f"moead_phase_{i}",
            )
        elif name == "vns":
            adapter_i = VNSAdapter(
                config=VNSConfig(
                    batch_size=int(max(4, vns_batch_size)),
                    k_max=int(max(1, vns_k_max)),
                    base_sigma=0.15,
                    scale=1.45,
                    max_sigma=2.0,
                    objective_aggregation="sum",
                ),
                name=f"vns_phase_{i}",
            )
        else:
            continue
        phases.append(SerialPhaseSpec(name=name, adapter=adapter_i, steps=int(max(1, st))))
        phase_meta.append({"name": str(name), "steps": int(max(1, st))})

    if not phases:
        phases = [
            SerialPhaseSpec(
                name="nsga2",
                adapter=NSGA2Adapter(
                    config=NSGA2Config(
                        population_size=int(pop),
                        offspring_size=int(pop),
                        crossover_rate=0.90,
                        objective_aggregation="sum",
                    )
                ),
                steps=int(gens),
            )
        ]
        phase_meta = [{"name": "nsga2", "steps": int(gens)}]

    adapter = StrategyChainAdapter(
        phases=phases,
        config=SerialStrategyConfig(repeat_last=False),
        name="portfolio_serial_chain",
    )
    return adapter, {
        "strategy": "portfolio",
        "max_generations": int(sum(int(v["steps"]) for v in phase_meta)),
        "portfolio_phases": phase_meta,
    }

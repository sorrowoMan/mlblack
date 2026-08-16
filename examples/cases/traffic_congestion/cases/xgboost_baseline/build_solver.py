import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from typing import Any, Mapping
from mlblack.assembly import build_trainer

try:
    from .pipeline import build_pipeline
except ImportError:
    from pipeline import build_pipeline

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "ci_interval_opt_table_no_flow_speed_occ_lag.csv"


def load_data():
    return build_pipeline(CSV_PATH)


def _resource_payload(resource_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(resource_context or {})
    if payload:
        return payload
    return {"device": "cpu", "threads": 1}


def build_solver(config=None, *, resource_context=None, component_overrides=None):
    """Canonical unified scaffold entry; returns the assembled Trainer."""

    payload = dict(config or {}) if isinstance(config, Mapping) else {}
    data = load_data()
    spec = {
        "preset": "xgboost",
        "resource_context": _resource_payload(resource_context),
        "params": {
            "population_size": int(payload.get("population_size", 8)),
            "mutation_scale": float(payload.get("mutation_scale", 0.2)),
        },
    }
    if component_overrides:
        spec.update(dict(component_overrides))
    trainer = build_trainer(spec, data)
    trainer.traffic_data = data
    return trainer

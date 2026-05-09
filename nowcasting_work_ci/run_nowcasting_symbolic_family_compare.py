from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CapabilitySpec, FlowAssemblySpec, NumericizerSpec, TrainerAssemblySpec
from core.common.contracts import ProcessedDataset
from examples.path_defaults import apply_env_defaults, default_work_ci_csv
from nowcasting_work_ci.mlblack_side.orthogonal_basis import _build_feature_bundle_from_reader
from workflow import SemanticTrainFlowSpec, TrainDataBundle, run_semantic_train_flow


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _build_bundle(args: argparse.Namespace) -> tuple[TrainDataBundle, dict[str, Any]]:
    feature_bundle, metadata = _build_feature_bundle_from_reader(
        csv_path=str(args.csv_path),
        target_col=str(args.target_col),
        test_fold_col=str(args.test_fold_col),
        lag_feature_enabled=bool(int(args.lag_feature_enabled)),
        lag_orders_csv=str(args.lag_orders),
        lag_sources_csv=str(args.lag_sources),
        lag_cross_enabled=bool(int(args.lag_cross_enabled)),
        lag_cross_quantiles_csv=str(args.lag_cross_quantiles),
        drop_same_day_flow_speed_occ=bool(int(args.drop_same_day_flow_speed_occ)),
        drop_feature_list_csv=str(args.drop_feature_list),
    )
    target_names = (str(args.target_col),)
    bundle = TrainDataBundle(
        train=ProcessedDataset(
            X_train=np.asarray(feature_bundle.X_train, dtype=float),
            y_train=np.asarray(feature_bundle.y_train, dtype=float),
            feature_names=tuple(feature_bundle.feature_names),
            target_names=target_names,
            metadata={
                "scenario": "nowcasting_work_ci",
                "split": "train",
                "feature_engineering": dict(metadata.get("feature_engineering", {})),
            },
        ),
        test=ProcessedDataset(
            X_train=np.asarray(feature_bundle.X_test, dtype=float),
            y_train=np.asarray(feature_bundle.y_test, dtype=float),
            feature_names=tuple(feature_bundle.feature_names),
            target_names=target_names,
            metadata={
                "scenario": "nowcasting_work_ci",
                "split": "test",
                "feature_engineering": dict(metadata.get("feature_engineering", {})),
            },
        ),
        metadata={
            **dict(metadata),
            "feature_names": tuple(str(value) for value in tuple(feature_bundle.feature_names)),
            "n_train": int(np.asarray(feature_bundle.X_train).shape[0]),
            "n_test": int(np.asarray(feature_bundle.X_test).shape[0]),
        },
    )
    return bundle, metadata


def _run_one(
    *,
    bundle: TrainDataBundle,
    trainer_params: dict[str, Any],
    run_name: str,
    output_dir: Path,
    db_path: str,
    namespace: str,
    tag: str,
) -> dict[str, Any]:
    spec = SemanticTrainFlowSpec(
        assembly=FlowAssemblySpec(
            trainer=TrainerAssemblySpec(
                trainer_key="symbolic",
                trainer_params=trainer_params,
            ),
            numericizer=NumericizerSpec(key="default", params={}),
            capabilities=(
                CapabilitySpec(
                    key="experiment_tracker",
                    params={
                        "db_path": str(db_path),
                        "namespace": str(namespace),
                        "tag": str(tag),
                        "io_mode": "batched",
                        "commit_interval": 0,
                    },
                ),
            ),
        ),
        eval_splits=("train", "test"),
        output_dir=str(output_dir),
        save_artifact=True,
        save_report=True,
        capability_strict=True,
        run_name=str(run_name),
    )
    result = run_semantic_train_flow(bundle, spec=spec)
    tracker = dict(result.report.get("experiment_tracker", {}))
    artifact_meta = dict(getattr(result.artifact, "metadata", {}) or {})
    schema = dict(artifact_meta.get("symbolic_artifact_schema", {}))
    basis_structure = dict(schema.get("basis_structure", {}))
    orthogonality = dict(basis_structure.get("orthogonality_status", {}))
    return {
        "run_name": str(run_name),
        "trainer_name": str(result.report.get("trainer_name") or getattr(result.artifact, "artifact_id", "")),
        "output_dir": str(result.output_dir),
        "metrics": _jsonable(result.metrics),
        "tracker": _jsonable(tracker),
        "artifact_id": str(getattr(result.artifact, "artifact_id", "")),
        "orthogonality_status": str(orthogonality.get("status", "")),
        "orthogonality_score": orthogonality.get("orthogonality_score"),
        "piecewise_gate_status": dict(schema.get("piecewise_gate_basis", {})).get("status"),
        "basis_scope": basis_structure.get("basis_scope"),
        "assembler_mode": dict(schema.get("assembler_structure", {})).get("assembler_mode"),
        "surface_key": str(tracker.get("surface_key", "")),
        "run_id": str(tracker.get("run_id", "")),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline symbolic vs orthogonal-basis-first symbolic on the engineered work_ci scenario and materialize both into the experiment DB.",
    )
    parser.add_argument("--csv-path", type=str, default=default_work_ci_csv())
    parser.add_argument("--target-col", type=str, default="ci")
    parser.add_argument("--test-fold-col", type=str, default="test_fold_10")
    parser.add_argument("--db-path", type=str, default=str(ROOT / "runs" / "experiments.sqlite3"))
    parser.add_argument("--namespace", type=str, default="nowcasting_work_ci_compare")
    parser.add_argument("--tag", type=str, default="symbolic_family_compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default=str(ROOT / "examples" / "out" / "nowcasting_symbolic_family_compare"))
    parser.add_argument("--lag-feature-enabled", type=int, default=1)
    parser.add_argument("--lag-orders", type=str, default="1,2,3")
    parser.add_argument("--lag-sources", type=str, default="ci,total_flow,avg_speed,avg_occ")
    parser.add_argument("--lag-cross-enabled", type=int, default=1)
    parser.add_argument("--lag-cross-quantiles", type=str, default="0.25,0.5,0.75")
    parser.add_argument("--drop-same-day-flow-speed-occ", type=int, default=1)
    parser.add_argument("--drop-feature-list", type=str, default="total_flow,avg_speed,avg_occ")
    parser.add_argument("--gate-feature-names", type=str, default="ci_lag1,avg_speed_lag1")
    parser.add_argument("--baseline-max-added-terms", type=int, default=6)
    parser.add_argument("--baseline-topk-features", type=int, default=10)
    parser.add_argument("--orth-candidate-limit", type=int, default=72)
    parser.add_argument("--orth-group-count", type=int, default=8)
    parser.add_argument("--orth-max-basis-count", type=int, default=6)
    parser.add_argument("--orth-selection-mode", type=str, default="interval_first")
    return parser


def main(argv: list[str] | None = None) -> None:
    apply_env_defaults()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() / stamp
    output_root.mkdir(parents=True, exist_ok=True)
    bundle, metadata = _build_bundle(args)
    gate_feature_names = tuple(
        item.strip()
        for item in str(args.gate_feature_names).split(",")
        if item.strip()
    )

    baseline_params = {
        "parameter_backend": "ridge",
        "task": "point",
        "structure_engine": {
            "structure_mode": "stagewise_search",
            "search_driver": "nsgablack",
            "dynamic_pool_enabled": True,
        },
        "artifact_id": f"nowcasting_symbolic_stagewise_{stamp}",
        "force_linear_base": "auto",
        "keep_search_trace": True,
        "search_max_added_terms": int(args.baseline_max_added_terms),
        "search_topk_features": int(args.baseline_topk_features),
        "search_max_pair_terms": 12,
        "search_max_candidates_per_iter": 256,
        "search_candidate_keep_top": 8,
        "search_include_hinge": True,
        "search_unary_ops": ("square", "sin", "cos", "tanh"),
        "search_online_beam_enabled": False,
        "search_path_memory_enabled": False,
        "search_graph_cache_enabled": False,
        "search_joint_bundle_enabled": False,
    }
    orthogonal_params = {
        "parameter_backend": "ridge",
        "task": "point",
        "structure_engine": {
            "structure_mode": "orthogonal_basis_search",
            "search_driver": "orthogonal_basis",
            "dynamic_pool_enabled": True,
            "metadata": {"supports_piecewise_basis": bool(gate_feature_names)},
        },
        "artifact_id": f"nowcasting_symbolic_orthogonal_{stamp}",
        "candidate_limit": int(args.orth_candidate_limit),
        "group_count": int(args.orth_group_count),
        "seed_candidate_count": 12,
        "min_basis_count": 3,
        "max_basis_count": int(args.orth_max_basis_count),
        "rolling_folds": 3,
        "rolling_val_ratio": 0.18,
        "min_train_ratio": 0.40,
        "selection_mode": str(args.orth_selection_mode),
        "random_seed": int(args.seed),
        "gate_feature_names": gate_feature_names,
        "enable_piecewise_basis": True,
        "search_graph_cache_enabled": False,
    }

    baseline = _run_one(
        bundle=bundle,
        trainer_params=baseline_params,
        run_name=f"{args.namespace}_baseline_{stamp}",
        output_dir=output_root / "baseline",
        db_path=str(args.db_path),
        namespace=str(args.namespace),
        tag=f"{args.tag}:baseline",
    )
    orthogonal = _run_one(
        bundle=bundle,
        trainer_params=orthogonal_params,
        run_name=f"{args.namespace}_orthogonal_{stamp}",
        output_dir=output_root / "orthogonal",
        db_path=str(args.db_path),
        namespace=str(args.namespace),
        tag=f"{args.tag}:orthogonal",
    )

    summary = {
        "generated_at": datetime.now().isoformat(),
        "db_path": str(args.db_path),
        "namespace": str(args.namespace),
        "tag": str(args.tag),
        "output_root": str(output_root),
        "feature_bundle_metadata": _jsonable(metadata),
        "bundle_metadata": _jsonable(bundle.metadata),
        "baseline": _jsonable(baseline),
        "orthogonal": _jsonable(orthogonal),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _metric(run: dict[str, Any], split: str, metric: str) -> str:
        value = dict(dict(run.get("metrics", {})).get(split, {})).get(metric)
        if value is None:
            return "nan"
        return f"{float(value):.6f}"

    print("NOWCASTING SYMBOLIC FAMILY COMPARE")
    print(f"summary={summary_path}")
    print(f"db_path={args.db_path}")
    print(
        "baseline   trainer={trainer} run_id={run_id} test_rmse={rmse} test_r2={r2}".format(
            trainer=baseline.get("trainer_name"),
            run_id=baseline.get("run_id"),
            rmse=_metric(baseline, "test", "rmse"),
            r2=_metric(baseline, "test", "r2"),
        )
    )
    print(
        "orthogonal trainer={trainer} run_id={run_id} test_rmse={rmse} test_r2={r2} orthogonality={orth}".format(
            trainer=orthogonal.get("trainer_name"),
            run_id=orthogonal.get("run_id"),
            rmse=_metric(orthogonal, "test", "rmse"),
            r2=_metric(orthogonal, "test", "r2"),
            orth=orthogonal.get("orthogonality_score"),
        )
    )


if __name__ == "__main__":
    main()

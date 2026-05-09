from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import create_default_config
from core.symbolic.symbolic_structure_search import (
    StructureSearchConfig,
    evaluate_genome_with_ridge,
    residual_guided_structure_search,
)
from examples.path_defaults import apply_env_defaults
from project.scaffold import ScaffoldSpec, _table_to_bundle, load_scaffold_spec, run_project_scaffold


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    return str(v)


def _run_variant(
    base: ScaffoldSpec,
    *,
    trainer_key: str,
    trainer_params: dict[str, Any],
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    train_spec = replace(
        base.train,
        trainer_key=str(trainer_key),
        trainer_params=dict(trainer_params),
        output_dir=str(output_dir),
        run_name=str(run_name),
    )
    spec = ScaffoldSpec(data=base.data, train=train_spec)

    t0 = time.perf_counter()
    result = run_project_scaffold(spec)
    elapsed = float(time.perf_counter() - t0)

    artifact = result.artifact
    model_meta = dict(getattr(artifact, "metadata", {}).get("model", {}))

    return {
        "status": "ok",
        "duration_sec": elapsed,
        "trainer_key": str(trainer_key),
        "output_dir": str(output_dir),
        "metrics": _jsonable(result.metrics),
        "model": _jsonable(model_meta),
        "formula_txt": str(output_dir / "artifact" / "formula.txt"),
        "flow_report": str(output_dir / "flow_report.json"),
    }


def _safe_run_variant(*args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return _run_variant(*args, **kwargs)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    apply_env_defaults()

    parser = argparse.ArgumentParser(description="Residual-guided structure search validation on mlblack scaffold.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "examples" / "configs" / "work_ci_symbolic_torch_v2.json"),
        help="Base scaffold config path.",
    )
    parser.add_argument("--max-added-terms", type=int, default=10)
    parser.add_argument("--topk-features", type=int, default=8)
    parser.add_argument("--max-pair-terms", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=220, help="Epochs for symbolic runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--output-root", type=str, default="")
    args = parser.parse_args()

    base_spec = load_scaffold_spec(args.config)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_root = Path(base_spec.train.output_dir).parent / f"{base_spec.train.run_name}_structure_search_{stamp}"
    out_root = Path(args.output_root).resolve() if str(args.output_root).strip() else default_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    bundle = _table_to_bundle(base_spec.data)
    train_ds = bundle.train
    test_ds = bundle.test
    if getattr(train_ds, "X_train", None) is None:
        raise TypeError("train split is not ProcessedDataset")
    if test_ds is None or getattr(test_ds, "X_train", None) is None:
        raise TypeError("test split is not ProcessedDataset")

    X_train = np.asarray(train_ds.X_train, dtype=float)
    y_train = np.asarray(train_ds.y_train, dtype=float)
    X_test = np.asarray(test_ds.X_train, dtype=float)
    y_test = np.asarray(test_ds.y_train, dtype=float)

    cfg = create_default_config()
    pipeline = cfg.pipelines.create(base_spec.train.pipeline_key, **dict(base_spec.train.pipeline_params))
    X_train_basis = np.asarray(pipeline.fit_transform(X_train, y_train), dtype=float)
    X_test_basis = np.asarray(pipeline.transform(X_test), dtype=float)

    search_cfg = StructureSearchConfig(
        max_added_terms=int(max(0, args.max_added_terms)),
        topk_features=int(max(1, args.topk_features)),
        max_pair_terms=int(max(0, args.max_pair_terms)),
    )

    t0 = time.perf_counter()
    search_res = residual_guided_structure_search(
        X_train_basis,
        y_train,
        feature_names=getattr(train_ds, "feature_names", None),
        config=search_cfg,
    )
    search_elapsed = float(time.perf_counter() - t0)

    seed_terms = int(X_train_basis.shape[1])
    seed_genome = tuple(search_res.genome[:seed_terms])

    ridge_seed = evaluate_genome_with_ridge(
        seed_genome,
        X_train=X_train_basis,
        y_train=y_train,
        X_eval=X_test_basis,
        y_eval=y_test,
        l2=float(search_cfg.ridge_l2),
    )
    ridge_final = evaluate_genome_with_ridge(
        search_res.genome,
        X_train=X_train_basis,
        y_train=y_train,
        X_eval=X_test_basis,
        y_eval=y_test,
        l2=float(search_cfg.ridge_l2),
    )

    search_payload = {
        "search_config": _jsonable(asdict(search_cfg)),
        "duration_sec": search_elapsed,
        "result": _jsonable(search_res.to_dict()),
        "ridge_eval": {
            "seed_metrics_train": _jsonable(ridge_seed.get("metrics_train")),
            "seed_metrics_test": _jsonable(ridge_seed.get("metrics_eval")),
            "final_metrics_train": _jsonable(ridge_final.get("metrics_train")),
            "final_metrics_test": _jsonable(ridge_final.get("metrics_eval")),
        },
    }
    (out_root / "structure_search.json").write_text(
        json.dumps(search_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    symbolic_base_params = dict(base_spec.train.trainer_params)
    symbolic_base_params["epochs"] = int(args.epochs)
    symbolic_base_params["device"] = "cpu"
    symbolic_base_params["random_seed"] = int(args.seed)

    baseline = _safe_run_variant(
        base_spec,
        trainer_key="symbolic_torch",
        trainer_params=symbolic_base_params,
        output_dir=out_root / "baseline_symbolic_v2",
        run_name="baseline_symbolic_v2",
    )

    searched_params = dict(symbolic_base_params)
    searched_params["genome"] = list(search_res.genome)
    searched_params["l1_readout"] = 0.0
    searched_params["l1_params"] = 0.0
    searched_params["lr"] = max(0.0012, float(searched_params.get("lr", 0.0008)))
    searched_params["weight_decay"] = min(1e-5, float(searched_params.get("weight_decay", 1e-4)))

    searched = _safe_run_variant(
        base_spec,
        trainer_key="symbolic_torch",
        trainer_params=searched_params,
        output_dir=out_root / "search_guided_symbolic",
        run_name="search_guided_symbolic",
    )

    xgboost_result: dict[str, Any] | None = None
    if not bool(args.skip_xgboost):
        xgb_params = {
            "n_estimators": 420,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "tree_method": "hist",
            "random_seed": int(args.seed),
        }
        xgboost_result = _safe_run_variant(
            base_spec,
            trainer_key="xgboost",
            trainer_params=xgb_params,
            output_dir=out_root / "baseline_xgboost",
            run_name="baseline_xgboost",
        )

    def _test_rmse(item: dict[str, Any] | None) -> float:
        if not isinstance(item, dict):
            return float("nan")
        try:
            return float(item["metrics"]["test"]["rmse"])
        except Exception:
            return float("nan")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": str(Path(args.config).resolve()),
        "output_root": str(out_root),
        "search": {
            "duration_sec": float(search_elapsed),
            "base_metrics_train_basis": _jsonable(search_res.base_metrics),
            "final_metrics_train_basis": _jsonable(search_res.final_metrics),
            "iterations": int(len(search_res.iterations)),
            "n_terms_final": int(len(search_res.genome)),
            "search_report": str(out_root / "structure_search.json"),
        },
        "ridge_eval": {
            "seed_metrics_train": _jsonable(ridge_seed.get("metrics_train")),
            "seed_metrics_test": _jsonable(ridge_seed.get("metrics_eval")),
            "final_metrics_train": _jsonable(ridge_final.get("metrics_train")),
            "final_metrics_test": _jsonable(ridge_final.get("metrics_eval")),
        },
        "runs": {
            "baseline_symbolic_v2": baseline,
            "search_guided_symbolic": searched,
            "baseline_xgboost": xgboost_result,
        },
        "delta": {
            "rmse_test_search_minus_baseline_symbolic": float(_test_rmse(searched) - _test_rmse(baseline)),
            "rmse_test_search_minus_xgboost": float(_test_rmse(searched) - _test_rmse(xgboost_result)),
            "rmse_test_ridge_final_minus_ridge_seed": (
                float(ridge_final["metrics_eval"]["rmse"] - ridge_seed["metrics_eval"]["rmse"])
                if isinstance(ridge_final.get("metrics_eval"), dict) and isinstance(ridge_seed.get("metrics_eval"), dict)
                else float("nan")
            ),
        },
    }

    (out_root / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("STRUCTURE SEARCH VALIDATION DONE")
    print(f"output_root={out_root}")
    print(f"search_iters={len(search_res.iterations)} n_terms={len(search_res.genome)}")
    print(f"baseline_symbolic_status={baseline.get('status')}")
    print(f"search_symbolic_status={searched.get('status')}")
    if isinstance(xgboost_result, dict):
        print(f"xgboost_status={xgboost_result.get('status')}")
    print(f"summary={out_root / 'summary.json'}")


if __name__ == "__main__":
    main()

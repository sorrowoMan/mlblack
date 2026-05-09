from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from core.artifacts.artifact import LinearSurrogateArtifact
from core.artifacts.piecewise_symbolic_interval_artifact import PiecewiseSymbolicIntervalSurrogateArtifact
from core.artifacts.sklearn_mlp_artifact import SklearnMLPSurrogateArtifact
from core.artifacts.symbolic_artifact import SymbolicSurrogateArtifact
from core.artifacts.symbolic_interval_artifact import SymbolicIntervalSurrogateArtifact
from core.artifacts.torch_artifact import TorchMLPSurrogateArtifact
from core.artifacts.xgboost_artifact import XGBoostSurrogateArtifact
from core.common.contracts import ProcessedDataset, SurrogateArtifact


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _artifact_loader_map():
    return {
        "linear": LinearSurrogateArtifact.load,
        "torch_mlp": TorchMLPSurrogateArtifact.load,
        "sklearn_mlp": SklearnMLPSurrogateArtifact.load,
        "xgboost": XGBoostSurrogateArtifact.load,
        "symbolic_torch": SymbolicSurrogateArtifact.load,
        "symbolic_torch_interval": SymbolicIntervalSurrogateArtifact.load,
        "piecewise_symbolic_torch_interval": PiecewiseSymbolicIntervalSurrogateArtifact.load,
    }


def _save_processed(path: Path, processed: ProcessedDataset) -> None:
    arrays: dict[str, np.ndarray] = {
        "X_train": np.asarray(processed.X_train, dtype=float),
        "y_train": np.asarray(processed.y_train, dtype=float),
    }
    if processed.X_valid is not None:
        arrays["X_valid"] = np.asarray(processed.X_valid, dtype=float)
    if processed.y_valid is not None:
        arrays["y_valid"] = np.asarray(processed.y_valid, dtype=float)
    if processed.X_test is not None:
        arrays["X_test"] = np.asarray(processed.X_test, dtype=float)
    if processed.y_test is not None:
        arrays["y_test"] = np.asarray(processed.y_test, dtype=float)
    np.savez(path / "processed.npz", **arrays)

    meta = {
        "feature_names": None if processed.feature_names is None else list(processed.feature_names),
        "target_names": None if processed.target_names is None else list(processed.target_names),
        "metadata": _jsonable(processed.metadata),
        "has_valid": bool(processed.X_valid is not None and processed.y_valid is not None),
        "has_test": bool(processed.X_test is not None and processed.y_test is not None),
    }
    (path / "processed_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_processed(path: Path) -> ProcessedDataset:
    arrays = np.load(path / "processed.npz")
    meta = json.loads((path / "processed_meta.json").read_text(encoding="utf-8"))

    return ProcessedDataset(
        X_train=np.asarray(arrays["X_train"], dtype=float),
        y_train=np.asarray(arrays["y_train"], dtype=float),
        X_valid=None if "X_valid" not in arrays else np.asarray(arrays["X_valid"], dtype=float),
        y_valid=None if "y_valid" not in arrays else np.asarray(arrays["y_valid"], dtype=float),
        X_test=None if "X_test" not in arrays else np.asarray(arrays["X_test"], dtype=float),
        y_test=None if "y_test" not in arrays else np.asarray(arrays["y_test"], dtype=float),
        feature_names=None
        if meta.get("feature_names") is None
        else tuple(str(v) for v in meta.get("feature_names", [])),
        target_names=None
        if meta.get("target_names") is None
        else tuple(str(v) for v in meta.get("target_names", [])),
        metadata=meta.get("metadata"),
    )


def _save_artifact(path: Path, artifact: SurrogateArtifact) -> dict[str, Any]:
    art_dir = path / "artifact"
    art_dir.mkdir(parents=True, exist_ok=True)
    artifact.save(str(art_dir))

    meta_path = art_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"checkpoint artifact meta missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "artifact_relpath": "artifact",
        "artifact_type": str(meta.get("artifact_type", "")),
    }


def _load_artifact(path: Path, artifact_type: str) -> SurrogateArtifact:
    loaders = _artifact_loader_map()
    key = str(artifact_type).strip().lower()
    loader = loaders.get(key)
    if loader is None:
        known = ", ".join(sorted(loaders.keys()))
        raise ValueError(f"Unknown artifact_type '{artifact_type}'. Known: [{known}]")
    return loader(str(path))


def save_train_checkpoint(
    *,
    checkpoint_dir: str,
    artifact: SurrogateArtifact,
    processed: ProcessedDataset,
    metrics: Mapping[str, Mapping[str, float]],
    report: Mapping[str, Any],
    run_name: str,
    output_dir: str | None,
) -> str:
    path = Path(checkpoint_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)

    artifact_info = _save_artifact(path, artifact)
    _save_processed(path, processed)

    (path / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path / "report.json").write_text(
        json.dumps(_jsonable(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": str(run_name),
        "output_dir": None if output_dir is None else str(output_dir),
        **artifact_info,
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(path)


def load_train_checkpoint(checkpoint_dir: str):
    path = Path(checkpoint_dir).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    report = json.loads((path / "report.json").read_text(encoding="utf-8"))
    processed = _load_processed(path)

    art_rel = str(manifest.get("artifact_relpath", "artifact"))
    art_type = str(manifest.get("artifact_type", ""))
    artifact = _load_artifact(path / art_rel, art_type)

    from core.orchestration.workflow import TrainFlowResult

    return TrainFlowResult(
        artifact=artifact,
        processed=processed,
        metrics=dict(metrics),
        report=dict(report),
        output_dir=manifest.get("output_dir"),
    )

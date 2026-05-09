from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class ArtifactPersistenceBase:
    """Shared persistence helpers for SurrogateArtifact implementations."""

    @staticmethod
    def _ensure_dir(out_dir: str) -> Path:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _save_npz(path: Path, filename: str, **arrays: Any) -> None:
        np.savez(path / str(filename), **arrays)

    @staticmethod
    def _load_npz(path: Path, filename: str):
        return np.load(path / str(filename))

    @staticmethod
    def _save_json(
        path: Path,
        filename: str,
        payload: Any,
        *,
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> None:
        (path / str(filename)).write_text(
            json.dumps(payload, ensure_ascii=bool(ensure_ascii), indent=int(indent)),
            encoding="utf-8",
        )

    @staticmethod
    def _load_json(path: Path, filename: str) -> Any:
        return json.loads((path / str(filename)).read_text(encoding="utf-8"))

    @staticmethod
    def _save_text(path: Path, filename: str, text: str) -> None:
        (path / str(filename)).write_text(str(text), encoding="utf-8")

    @staticmethod
    def _load_text(path: Path, filename: str) -> str:
        return (path / str(filename)).read_text(encoding="utf-8")

    @staticmethod
    def _save_meta(path: Path, meta: Mapping[str, Any]) -> None:
        ArtifactPersistenceBase._save_json(path, "meta.json", dict(meta), ensure_ascii=False, indent=2)

    @staticmethod
    def _load_meta(path: Path) -> dict[str, Any]:
        return dict(ArtifactPersistenceBase._load_json(path, "meta.json"))

    @staticmethod
    def _save_pickle(path: Path, filename: str, obj: Any) -> None:
        with (path / str(filename)).open("wb") as f:
            pickle.dump(obj, f)

    @staticmethod
    def _load_pickle(path: Path, filename: str) -> Any:
        with (path / str(filename)).open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _save_torch(path: Path, filename: str, payload: Any) -> None:
        import torch

        torch.save(payload, path / str(filename))

    @staticmethod
    def _load_torch(path: Path, filename: str, *, map_location: str = "cpu") -> Any:
        import torch

        return torch.load(path / str(filename), map_location=map_location)

    def _common_meta(self, *, artifact_type: str | None = None, **extra: Any) -> dict[str, Any]:
        out = {
            "artifact_id": str(getattr(self, "artifact_id", "")),
            "feature_names": list(getattr(self, "feature_names", ()) or ()),
            "target_names": list(getattr(self, "target_names", ()) or ()),
            "pipeline_name": str(getattr(self, "pipeline_name", "identity")),
            "pipeline_state": dict(getattr(self, "pipeline_state", {}) or {}),
            "ood_z_threshold": float(getattr(self, "ood_z_threshold", 4.0)),
            "metadata": dict(getattr(self, "metadata", {}) or {}),
        }
        if artifact_type:
            out["artifact_type"] = str(artifact_type)
        out.update(dict(extra))
        return out

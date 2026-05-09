from __future__ import annotations

import sys
from pathlib import Path


def default_repo_roots() -> tuple[Path, Path]:
    package_root = Path(__file__).resolve().parents[1]
    mlblack_root = package_root.parent
    nsgablack_root = mlblack_root.parent / "nsgablack"
    return mlblack_root, nsgablack_root


def ensure_repo_import_order(
    *,
    mlblack_root: Path | None = None,
    nsgablack_root: Path | None = None,
) -> tuple[Path, Path]:
    ml_root, ns_root = default_repo_roots()
    ml_root = Path(mlblack_root).resolve() if mlblack_root is not None else ml_root
    ns_root = Path(nsgablack_root).resolve() if nsgablack_root is not None else ns_root

    ordered: list[Path] = [ml_root]
    if ns_root.exists():
        ordered.append(ns_root)

    normalized_targets = {str(path.resolve()).lower() for path in ordered}
    preserved: list[str] = []
    for raw in sys.path:
        try:
            normalized = str(Path(raw).resolve()).lower()
        except Exception:
            normalized = str(raw).strip().lower()
        if normalized in normalized_targets:
            continue
        preserved.append(raw)

    sys.path[:] = [str(path) for path in ordered] + preserved
    return ml_root, ns_root


__all__ = ["default_repo_roots", "ensure_repo_import_order"]

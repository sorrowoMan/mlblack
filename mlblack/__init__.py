from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    try:
        return version("mlblack")
    except PackageNotFoundError:
        return "0.0.0"


__version__ = _resolve_version()

__all__ = ["__version__"]


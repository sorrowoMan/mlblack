"""Case-level pipeline package.

Primary entry: pipeline/main.py:build_pipeline
"""

try:
    from .main import build_pipeline, run_pipeline_slot
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.main":
        raise
    __all__ = []
else:
    __all__ = ["build_pipeline", "run_pipeline_slot"]

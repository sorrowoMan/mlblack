from __future__ import annotations

from typing import Any

from .contracts import (
    RUN_SURFACE_CONTRACT_VERSION,
    ArtifactRecord,
    AssemblyRecord,
    RunRecord,
    SurfaceRecord,
    make_artifact_record,
    make_assembly_record,
    make_run_record,
    make_surface_record,
)


def dashboard_main(*args: Any, **kwargs: Any):
    from .dashboard import main

    return main(*args, **kwargs)


def build_streamlit_command(*args: Any, **kwargs: Any):
    from .dashboard import build_streamlit_command as _build_streamlit_command

    return _build_streamlit_command(*args, **kwargs)


def dashboard_script_path():
    from .dashboard import dashboard_script_path as _dashboard_script_path

    return _dashboard_script_path()


__all__ = [
    "RUN_SURFACE_CONTRACT_VERSION",
    "SurfaceRecord",
    "AssemblyRecord",
    "ArtifactRecord",
    "RunRecord",
    "make_surface_record",
    "make_assembly_record",
    "make_artifact_record",
    "make_run_record",
    "build_streamlit_command",
    "dashboard_script_path",
    "dashboard_main",
]

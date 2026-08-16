"""mlblack semantic entry over the shared blackbase Project runner."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from blackbase.project.project_runner import build_parser
from blackbase.project.project_runner import execute_project as _blackbase_execute_project
from blackbase.project.project_runner import main as _blackbase_main
from blackbase.project.project_runner import run_project as _blackbase_run_project
from blackbase.project.execution import ProjectRunResult

_MLBLACK_PARENT = Path(__file__).resolve().parent.parent.parent


def run_project(
    project_root: Path | str,
    *,
    group: str = "default",
    check: bool = False,
    build_check: bool = False,
    case_args: Sequence[str] | None = None,
    record: bool = True,
    run_id: str | None = None,
    resume_from: Path | str | None = None,
) -> int:
    return _blackbase_run_project(
        project_root,
        group=group,
        check=check,
        build_check=build_check,
        case_args=case_args,
        record=record,
        run_id=run_id,
        resume_from=resume_from,
        framework="mlblack",
        resource_env_var="MLBLACK_RESOURCE_CONTEXT_JSON",
        extra_python_paths=(_MLBLACK_PARENT,),
    )


def execute_project(
    project_root: Path | str,
    *,
    group: str = "default",
    check: bool = False,
    build_check: bool = False,
    case_args: Sequence[str] | None = None,
    record: bool = True,
    run_id: str | None = None,
    resume_from: Path | str | None = None,
) -> ProjectRunResult:
    """Execute a Project and retain structured Case results and artifact refs."""

    return _blackbase_execute_project(
        project_root,
        group=group,
        check=check,
        build_check=build_check,
        case_args=case_args,
        record=record,
        run_id=run_id,
        resume_from=resume_from,
        framework="mlblack",
        resource_env_var="MLBLACK_RESOURCE_CONTEXT_JSON",
        extra_python_paths=(_MLBLACK_PARENT,),
    )


def main(project_root: Path | str | None = None, argv: Sequence[str] | None = None) -> int:
    return _blackbase_main(
        project_root,
        argv,
        framework="mlblack",
        resource_env_var="MLBLACK_RESOURCE_CONTEXT_JSON",
        extra_python_paths=(_MLBLACK_PARENT,),
    )


__all__ = ["build_parser", "execute_project", "main", "run_project"]

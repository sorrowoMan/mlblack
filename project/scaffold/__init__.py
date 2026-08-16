"""mlblack semantic templates over the shared blackbase scaffold substrate."""

from __future__ import annotations

from pathlib import Path

from blackbase.project.scaffold import add_component
from blackbase.project.scaffold import add_case as _add_case
from blackbase.project.scaffold import create_project as _create_project
from blackbase.project.check_output import build_case_check_payload, format_case_check, print_case_check

_SCAFFOLD_ROOT = Path(__file__).resolve().parent
_TEMPLATE_BY_KIND = {
    "solver": _SCAFFOLD_ROOT / "solver_case_template",
    "trainer": _SCAFFOLD_ROOT / "trainer_case_template",
}


def create_project(project_name: str | Path, *, force: bool = False):
    return _create_project(
        project_name,
        force=bool(force),
        framework="mlblack",
        project_template=None,
    )


def add_case(
    case_name: str,
    case_type: str = "trainer",
    *,
    framework: str = "mlblack",
    project_root: str | Path | None = None,
):
    return _add_case(
        case_name,
        case_type,
        framework=str(framework or "mlblack"),
        project_root=project_root,
        template_by_kind=_TEMPLATE_BY_KIND,
    )


init_project = create_project
create_standard_scaffold = create_project

__all__ = [
    "add_case",
    "add_component",
    "build_case_check_payload",
    "create_project",
    "create_standard_scaffold",
    "format_case_check",
    "init_project",
    "print_case_check",
]

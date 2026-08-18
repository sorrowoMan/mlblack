"""mlblack semantic templates over the shared blackbase scaffold substrate."""

from __future__ import annotations

from pathlib import Path

from blackbase.project.scaffold import add_component as _add_component
from blackbase.project.scaffold import add_case as _add_case
from blackbase.project.scaffold import create_project as _create_project

from .component_templates import render_component_template
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


def add_component(
    component_name: str,
    component_kind: str,
    *,
    case_name: str | None = None,
    slot: str | None = None,
    project_root: str | Path | None = None,
):
    return _add_component(
        component_name,
        component_kind,
        case_name=case_name,
        slot=slot,
        project_root=project_root,
        framework="mlblack",
        template_providers={"mlblack": render_component_template},
    )


__all__ = [
    "add_case",
    "add_component",
    "build_case_check_payload",
    "create_project",
    "format_case_check",
    "print_case_check",
]

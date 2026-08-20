from __future__ import annotations

from pathlib import Path

import pytest

from blackbase.project import build_case, load_case_builder
from blackbase.project.runtime import close_case_after_build_check
from blackbase.resources import ResourceContext


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("project_name", "case_name", "case_kind"),
    (
        ("cross_framework", "cross_framework", "solver"),
        ("cross_framework", "inner_training", "trainer"),
        ("benchmarks", "benchmark_tiny_cnn_classification", "trainer"),
        ("benchmarks", "benchmark_tiny_gnn_classification", "trainer"),
        ("benchmarks", "benchmark_tiny_cnn_contrastive", "trainer"),
        ("benchmarks", "benchmark_tiny_transformer_lm", "trainer"),
    ),
)
def test_canonical_cases_accept_authoritative_resource_rebinding(
    project_name: str,
    case_name: str,
    case_kind: str,
) -> None:
    project_root = _REPO_ROOT / "examples" / "cases" / project_name
    builder = load_case_builder(project_root, case_name, case_kind=case_kind)
    grant = {
        "scope": "training",
        "threads": 2,
        "namespace": f"test.{project_name}.{case_name}",
        "grant": {"threads": 2, "workers": 1},
    }

    runner = build_case(
        builder,
        resource_context=grant,
        component_overrides={},
    )

    assert (
        ResourceContext.from_mapping(runner.resource_context).as_dict()
        == ResourceContext.from_mapping(grant).as_dict()
    )
    assert runner.resource_binding_audit["current"] is True
    assert runner.resource_binding_audit["method"] == "set_resource_context"
    close_case_after_build_check(runner)


@pytest.mark.parametrize("app_name", ("catalog_assistant", "mall_ai_assistant"))
def test_application_examples_are_not_disguised_as_cases(app_name: str) -> None:
    app_root = _REPO_ROOT / "examples" / "apps" / app_name
    former_project = _REPO_ROOT / "examples" / "cases" / app_name

    assert (app_root / "server.py").is_file()
    assert not (app_root / ".case").exists()
    assert not (former_project / "project_config.py").exists()
    assert not (former_project / "run_project.py").exists()

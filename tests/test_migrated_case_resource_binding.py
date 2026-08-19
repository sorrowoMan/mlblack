from __future__ import annotations

from pathlib import Path

import pytest

from blackbase.project import build_case, load_case_builder
from blackbase.resources import ResourceContext


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "case_name",
    ("cross_framework", "catalog_assistant", "mall_ai_assistant", "benchmarks"),
)
def test_migrated_runner_accepts_authoritative_resource_rebinding(case_name: str) -> None:
    project_root = _REPO_ROOT / "examples" / "cases" / case_name
    builder = load_case_builder(project_root, case_name, case_kind="trainer")
    grant = {
        "scope": "training",
        "threads": 2,
        "namespace": f"test.{case_name}",
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

from __future__ import annotations

import nowcasting_work_ci.mlblack_side.runtime.workflow as runtime_workflow

from nowcasting_work_ci.mlblack_side.runtime.contracts import (
    RUNTIME_LAYER_RULES,
    RUNTIME_STAGE_CONTRACTS,
    RuntimeContextKey,
    RuntimeStageName,
)
from nowcasting_work_ci.mlblack_side.runtime.stages import build_experiment_stages


def test_stage_builder_matches_declared_runtime_contracts() -> None:
    built_stage_names = [stage.name for stage in build_experiment_stages([])]
    declared_stage_names = [contract.name.value for contract in RUNTIME_STAGE_CONTRACTS]

    assert built_stage_names == declared_stage_names
    assert all(contract.allows_direct_io is False for contract in RUNTIME_STAGE_CONTRACTS)


def test_runtime_layer_rules_keep_side_effects_in_plugins_only() -> None:
    rules = {rule.layer: rule for rule in RUNTIME_LAYER_RULES}

    assert rules["runtime/actions/*"].allows_direct_io is False
    assert rules["plugins/*"].allows_direct_io is True
    assert "summary write" in rules["runtime/actions/*"].forbidden
    assert "resource close" in rules["runtime/actions/*"].forbidden
    assert RuntimeContextKey.RUNTIME_SEED.value == "runtime_seed"
    assert RuntimeStageName.ASSEMBLE_RESULT.value == "assemble_result"


def test_runtime_workflow_preserves_cli_args_when_argv_is_omitted(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(self, stages, *, context=None):  # type: ignore[no-untyped-def]
        captured["stage_names"] = [stage.name for stage in stages]
        captured["context"] = dict(context or {})
        return {"status": "ok"}

    monkeypatch.setattr(runtime_workflow.ExperimentOrchestrator, "run", _fake_run)
    monkeypatch.setattr(runtime_workflow.sys, "argv", ["run.py", "--seed", "77", "--generations", "3"])

    result = runtime_workflow.main(argv=None, enable_default_plugins=False)

    assert result == {"status": "ok"}
    assert captured["context"][RuntimeContextKey.ARGV.value] == ["--seed", "77", "--generations", "3"]
    assert captured["context"][RuntimeContextKey.RUNTIME_SEED.value] == 77

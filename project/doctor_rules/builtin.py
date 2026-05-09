from __future__ import annotations

import re
from pathlib import Path

from catalog import list_entries
from project.doctor_types import DoctorProblem, DoctorRule

_ABS_PATH_PATTERN = re.compile(r"C:\\Users\\[^\\]+\\Desktop\\", re.IGNORECASE)


def _check_required_files(root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    for rel in ("README.md", "pyproject.toml"):
        p = root / rel
        if not p.exists():
            out.append(
                DoctorProblem(
                    severity="error",
                    code="required_file_missing",
                    message=f"Missing required file: {rel}",
                    path=str(p),
                )
            )
    return out


def _check_tests(root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    tests_dir = root / "tests"
    if not tests_dir.exists():
        out.append(
            DoctorProblem(
                severity="error",
                code="tests_missing",
                message="tests directory does not exist",
                path=str(tests_dir),
            )
        )
        return out

    py_tests = list(tests_dir.rglob("test_*.py"))
    if not py_tests:
        out.append(
            DoctorProblem(
                severity="error",
                code="tests_empty",
                message="No test_*.py files found under tests/",
                path=str(tests_dir),
            )
        )
    return out


def _check_catalog_profiles(_root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    framework_core = list_entries(profile="framework-core")
    leaked = [e for e in framework_core if e.kind in {"doc", "example"}]
    if leaked:
        out.append(
            DoctorProblem(
                severity="error",
                code="catalog_profile_leak",
                message=f"framework-core profile leaked non-core entries: {[x.key for x in leaked[:5]]}",
            )
        )

    default = list_entries(profile="default")
    if len(default) < len(framework_core):
        out.append(
            DoctorProblem(
                severity="error",
                code="catalog_profile_size_invalid",
                message="default profile should not be smaller than framework-core profile",
            )
        )
    return out


def _check_absolute_paths(root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    ex_root = root / "examples"
    if not ex_root.exists():
        return out

    for path in sorted(ex_root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("examples/out/"):
            continue
        if rel == "examples/path_defaults.py":
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        if _ABS_PATH_PATTERN.search(text):
            out.append(
                DoctorProblem(
                    severity="error",
                    code="absolute_path_literal",
                    message="Hard-coded absolute Desktop path found; use path_defaults/env instead",
                    path=str(path),
                )
            )
    return out


def _check_workspace_noise(root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []

    temp_hits: list[str] = []
    for pattern in (".tmp*.log", "tmp_*.txt"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                temp_hits.append(path.name)
    if temp_hits:
        out.append(
            DoctorProblem(
                severity="warning",
                code="root_temp_files_present",
                message=f"Root temp files should be cleaned or moved out of the framework root: {temp_hits[:6]}",
                path=str(root),
            )
        )

    cache_hits: list[str] = []
    for rel in (".pytest_cache", ".mlblack_cache"):
        path = root / rel
        if path.exists():
            cache_hits.append(rel)
    if cache_hits:
        out.append(
            DoctorProblem(
                severity="warning",
                code="root_cache_dirs_present",
                message=f"Root cache directories are present: {cache_hits}",
                path=str(root),
            )
        )

    pycache_hits: list[str] = []
    for path in sorted(root.rglob("__pycache__")):
        try:
            rel = path.relative_to(root).as_posix()
        except Exception:
            rel = str(path)
        if rel.startswith("examples/out/") or rel.startswith("_scenario_runs/"):
            continue
        pycache_hits.append(rel)
    if pycache_hits:
        out.append(
            DoctorProblem(
                severity="warning",
                code="pycache_dirs_present",
                message=f"Python bytecode caches should be cleaned before packaging: {pycache_hits[:8]}",
                path=str(root),
            )
        )

    return out


def _check_root_surface(root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    allowed_root_markdown = {
        "README.md",
        "AGENTS.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md",
    }
    leaked_docs = [
        path.name
        for path in sorted(root.glob("*.md"))
        if path.name not in allowed_root_markdown
    ]
    if leaked_docs:
        out.append(
            DoctorProblem(
                severity="error",
                code="root_non_public_docs_present",
                message=f"Top-level non-public markdown files should live under docs/: {leaked_docs[:8]}",
                path=str(root),
            )
        )
    return out


def _check_catalog_family_route_contracts(_root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    formal_presets = [
        entry
        for entry in list_entries(profile="framework-core", kind="preset")
        if str(dict(entry.fields).get("surface_status", "")).strip().lower() == "formal"
    ]
    malformed: list[str] = []
    for entry in formal_presets:
        fields = dict(entry.fields)
        route_keys = tuple(fields.get("family_route_keys", ()))
        formal_preset = str(fields.get("family_route_formal_preset", "")).strip()
        route_match_fields = tuple(fields.get("family_route_match_fields", ()))
        route_statuses = tuple(fields.get("family_route_statuses", ()))
        if not route_keys or not formal_preset or not route_match_fields or not route_statuses:
            malformed.append(str(entry.key))
    if malformed:
        out.append(
            DoctorProblem(
                severity="error",
                code="catalog_family_route_contract_incomplete",
                message=(
                    "Formal preset entries must expose family router contract fields "
                    f"(family_route_keys/formal_preset/match_fields/statuses): {malformed[:8]}"
                ),
            )
        )
    return out


def _check_catalog_mount_contracts(_root: Path) -> list[DoctorProblem]:
    out: list[DoctorProblem] = []
    required_fields_by_kind = {
        "component": ("mount_plane", "mount_point", "orchestration_phases", "contract_consumes", "contract_provides", "contract_mutates"),
        "provider": ("mount_plane", "mount_point", "orchestration_phases", "contract_consumes", "contract_provides", "contract_mutates"),
        "plugin": ("mount_plane", "mount_point", "orchestration_phases", "contract_consumes", "contract_provides", "contract_mutates", "contract_cache"),
    }
    malformed: list[str] = []
    for kind, required_fields in required_fields_by_kind.items():
        for entry in list_entries(profile="framework-core", kind=kind):
            fields = dict(entry.fields)
            missing = [field for field in required_fields if field not in fields]
            empty = [
                field
                for field in ("mount_plane", "mount_point", "orchestration_phases")
                if field in fields and not fields.get(field)
            ]
            if missing or empty:
                detail = f"{entry.key}"
                if missing:
                    detail += f" missing={missing}"
                if empty:
                    detail += f" empty={empty}"
                malformed.append(detail)
    if malformed:
        out.append(
            DoctorProblem(
                severity="error",
                code="catalog_mount_contract_incomplete",
                message=(
                    "component/provider/plugin catalog entries must declare complete mount contracts: "
                    f"{malformed[:8]}"
                ),
            )
        )
    return out


def register_rules() -> tuple[DoctorRule, ...]:
    return (
        DoctorRule(
            rule_id="required_files",
            description="Check required root files exist",
            run=_check_required_files,
        ),
        DoctorRule(
            rule_id="tests_present",
            description="Check tests directory and test files",
            run=_check_tests,
        ),
        DoctorRule(
            rule_id="catalog_profiles",
            description="Validate default/framework-core profile invariants",
            run=_check_catalog_profiles,
        ),
        DoctorRule(
            rule_id="absolute_paths",
            description="Disallow hard-coded Desktop absolute paths in examples",
            run=_check_absolute_paths,
        ),
        DoctorRule(
            rule_id="workspace_noise",
            description="Warn on root temp files and cache directories that should not clutter the framework root",
            run=_check_workspace_noise,
        ),
        DoctorRule(
            rule_id="root_surface",
            description="Disallow non-public markdown documents at the framework root",
            run=_check_root_surface,
        ),
        DoctorRule(
            rule_id="catalog_family_route_contracts",
            description="Require formal preset entries to expose complete family router contract fields",
            run=_check_catalog_family_route_contracts,
        ),
        DoctorRule(
            rule_id="catalog_mount_contracts",
            description="Require component/provider/plugin entries to expose complete mount contracts",
            run=_check_catalog_mount_contracts,
        ),
    )

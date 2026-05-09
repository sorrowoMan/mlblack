from __future__ import annotations

import gc
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from catalog.registry import CatalogEntry
from project.doctor import run_doctor


class TestDoctor(unittest.TestCase):
    def test_doctor_has_no_error_on_repo_root(self) -> None:
        root = Path(__file__).resolve().parents[1]
        problems = run_doctor(root)
        errors = [p for p in problems if str(p.severity).lower() == "error"]
        self.assertEqual(errors, [])

    def test_doctor_rejects_non_public_root_markdown(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="mlblack_doctor_root_surface_"))
        try:
            (tmp_dir / "README.md").write_text("# demo\n", encoding="utf-8")
            (tmp_dir / "pyproject.toml").write_text("[build-system]\nrequires=[]\n", encoding="utf-8")
            tests_dir = tmp_dir / "tests"
            tests_dir.mkdir(parents=True, exist_ok=True)
            (tests_dir / "test_smoke.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")
            (tmp_dir / "internal_notes.md").write_text("move me to docs\n", encoding="utf-8")

            problems = run_doctor(tmp_dir)
            codes = {str(problem.code) for problem in problems}
            self.assertIn("root_non_public_docs_present", codes)
        finally:
            gc.collect()
            time.sleep(0.05)
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_doctor_rejects_incomplete_mount_contract_entries(self) -> None:
        broken_component = CatalogEntry(
            key="component:broken",
            kind="component",
            name="broken",
            source="unit",
            fields={
                "mount_plane": "trainer",
                "mount_point": "",
                "orchestration_phases": tuple(),
                "contract_consumes": tuple(),
                "contract_provides": tuple(),
                "contract_mutates": tuple(),
            },
        )
        with patch("project.doctor_rules.builtin.list_entries") as mocked:
            mocked.side_effect = lambda *args, **kwargs: (broken_component,) if kwargs.get("kind") == "component" else tuple()
            problems = run_doctor(Path(__file__).resolve().parents[1])
        codes = {str(problem.code) for problem in problems}
        self.assertIn("catalog_mount_contract_incomplete", codes)

    def test_doctor_rejects_incomplete_family_route_contract_entries(self) -> None:
        broken_preset = CatalogEntry(
            key="preset:tree_ensemble",
            kind="preset",
            name="tree_ensemble",
            source="unit",
            fields={
                "surface_status": "formal",
                "family_route_keys": tuple(),
                "family_route_formal_preset": "",
                "family_route_match_fields": tuple(),
                "family_route_statuses": tuple(),
            },
        )
        with patch("project.doctor_rules.builtin.list_entries") as mocked:
            mocked.side_effect = lambda *args, **kwargs: (broken_preset,) if kwargs.get("kind") == "preset" else tuple()
            problems = run_doctor(Path(__file__).resolve().parents[1])
        codes = {str(problem.code) for problem in problems}
        self.assertIn("catalog_family_route_contract_incomplete", codes)


if __name__ == "__main__":
    unittest.main()

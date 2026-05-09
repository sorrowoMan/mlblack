from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from catalog.project_catalog import load_project_entries
from project.scaffold import init_project


class TestScaffoldTemplate(unittest.TestCase):
    def test_generated_assembly_supports_new_and_legacy_workflow_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "scaffold_case"
            init_project(target, force=True)

            assembly_path = target / "assembly.py"
            text = assembly_path.read_text(encoding="utf-8")

            self.assertIn('core" / "orchestration" / "workflow.py', text)
            self.assertIn('or (path / "core" / "workflow.py").exists()', text)

            exec_schema_path = target / "schema" / "execution_schema.py"
            self.assertTrue(exec_schema_path.exists())
            self.assertIn("EXECUTION_SPEC_SCHEMA", exec_schema_path.read_text(encoding="utf-8"))

            trainer_contracts_path = target / "schema" / "trainer_contracts.py"
            self.assertTrue(trainer_contracts_path.exists())
            trainer_contracts_text = trainer_contracts_path.read_text(encoding="utf-8")
            self.assertIn("TRAINER_CONTRACTS", trainer_contracts_text)
            self.assertIn("TRAINER_RESOURCE_PROFILES", trainer_contracts_text)

            config_text = (target / "config.py").read_text(encoding="utf-8")
            self.assertIn("from schema.execution_schema import EXECUTION_SPEC_SCHEMA", config_text)
            self.assertIn("from schema.trainer_contracts import TRAINER_CONTRACTS, TRAINER_RESOURCE_PROFILES", config_text)
            self.assertIn("EXECUTION_BACKEND_CHOICES", config_text)
            self.assertIn("TRAINER_CONTRACT_KEYS", config_text)
            self.assertIn("TRAINER_PRESET_RESOURCE_PROFILES", config_text)

            readme_text = (target / "README.md").read_text(encoding="utf-8")
            self.assertIn("schema/trainer_contracts.py", readme_text)
            self.assertIn("catalog/entries.toml", readme_text)

            catalog_entries_path = target / "catalog" / "entries.toml"
            self.assertTrue(catalog_entries_path.exists())
            catalog_entries_text = catalog_entries_path.read_text(encoding="utf-8")
            self.assertIn("[[entry]]", catalog_entries_text)
            self.assertIn('key = "preset:local_demo"', catalog_entries_text)

            project_entries = load_project_entries(target)
            self.assertTrue(any(entry.key == "preset:local_demo" for entry in project_entries))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from project.doctor import run_doctor


class TestDoctorPlugins(unittest.TestCase):
    def test_external_plugin_rule_is_loaded(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            rules_dir = Path(tmp)
            plugin_path = rules_dir / "custom_ping.py"
            plugin_path.write_text(
                textwrap.dedent(
                    """
                    from project.doctor_types import DoctorProblem, DoctorRule

                    def _run(root):
                        return [DoctorProblem(severity="warning", code="custom_ping", message="ok", path=str(root))]

                    def register_rules():
                        return (DoctorRule(rule_id="custom_ping_rule", description="custom ping", run=_run),)
                    """
                ),
                encoding="utf-8",
            )

            problems = run_doctor(
                root,
                rules_dir=rules_dir,
                only_rule_ids=["custom_ping_rule"],
            )
            self.assertEqual(len(problems), 1)
            self.assertEqual(problems[0].code, "custom_ping")
            self.assertEqual(str(problems[0].severity).lower(), "warning")


if __name__ == "__main__":
    unittest.main()

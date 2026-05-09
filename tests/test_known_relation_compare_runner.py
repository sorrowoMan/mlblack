from __future__ import annotations

import unittest

from examples.run_symbolic_known_relation_compare import _resolve_orthogonal_trainer_overrides


class TestKnownRelationCompareRunner(unittest.TestCase):
    def test_resolve_orthogonal_trainer_overrides_strips_orth_prefix(self) -> None:
        overrides = _resolve_orthogonal_trainer_overrides(
            {
                "orchestrator_hints": {
                    "trainer_params_overrides": {
                        "orth_mechanistic_feature_groups": (("activation_energy", "temperature"),),
                        "orth_gate_candidate_screen_reserve": 3,
                        "orth_selection_mode": "rmse_first",
                        "cross_explanatory_rejection_mode": "proxy_group_hard",
                    }
                }
            }
        )

        self.assertEqual(
            tuple(tuple(group) for group in tuple(overrides.get("mechanistic_feature_groups", ()))),
            (("activation_energy", "temperature"),),
        )
        self.assertEqual(int(overrides.get("gate_candidate_screen_reserve", 0)), 3)
        self.assertEqual(str(overrides.get("selection_mode") or ""), "rmse_first")
        self.assertEqual(
            str(overrides.get("cross_explanatory_rejection_mode") or ""),
            "proxy_group_hard",
        )


if __name__ == "__main__":
    unittest.main()

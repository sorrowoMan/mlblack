from __future__ import annotations

import gc
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from project.scaffold import build_scaffold_spec, run_project_scaffold


class TestDataSourcesCompose(unittest.TestCase):
    def test_scaffold_runs_with_composed_csv_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 90
            dates = pd.date_range("2024-01-01", periods=n, freq="D").astype(str)
            rng = np.random.default_rng(11)

            main = pd.DataFrame(
                {
                    "date": dates,
                    "target": 0.0,
                    "base_x": rng.normal(size=n),
                }
            )
            traffic = pd.DataFrame(
                {
                    "date": dates,
                    "speed": rng.normal(loc=60.0, scale=10.0, size=n),
                    "flow": rng.normal(loc=1200.0, scale=200.0, size=n),
                }
            )
            weather = pd.DataFrame(
                {
                    "date": dates,
                    "aqi": rng.normal(loc=90.0, scale=15.0, size=n),
                }
            )
            main["target"] = (
                0.3 * main["base_x"] - 0.02 * traffic["speed"] + 0.001 * traffic["flow"] + 0.01 * weather["aqi"]
            )

            p_main = root / "main.csv"
            p_traffic = root / "traffic.csv"
            p_weather = root / "weather.csv"
            main.to_csv(p_main, index=False)
            traffic.to_csv(p_traffic, index=False)
            weather.to_csv(p_weather, index=False)

            payload = {
                "data": {
                    "sources": [
                        {"name": "main", "kind": "csv", "path": str(p_main)},
                        {"name": "traffic", "kind": "csv", "path": str(p_traffic), "prefix": "tr_"},
                        {"name": "weather", "kind": "csv", "path": str(p_weather), "prefix": "wx_"},
                    ],
                    "merge_on": ["date"],
                    "merge_how": "inner",
                    "target_col": "target",
                    "date_col": "date",
                    "feature_recipe": "raw_all_numeric",
                    "split_mode": "ratio",
                    "test_ratio": 0.2,
                    "random_seed": 42,
                },
                "train": {
                    "trainer_key": "ridge",
                    "trainer_params": {"l2": 0.0},
                    "run_name": "compose_csv_test",
                    "output_dir": str(root / "runs_csv"),
                    "state_backend": {
                        "context": {"backend": "memory"},
                        "snapshot": {"backend": "memory"},
                    },
                },
            }

            spec = build_scaffold_spec(payload)
            result = run_project_scaffold(spec)
            self.assertIn("train", result.metrics)
            self.assertIn("test", result.metrics)

            meta = dict(result.report.get("data", {}).get("metadata", {}))
            self.assertEqual(str(meta.get("source_mode")), "composed")
            self.assertEqual(len(tuple(meta.get("data_sources", ()))), 3)
            feature_names = tuple(result.processed.feature_names or ())
            self.assertIn("tr_speed", feature_names)
            self.assertIn("wx_aqi", feature_names)

    def test_scaffold_runs_with_sqlite_and_csv_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            n = 64
            dates = pd.date_range("2024-03-01", periods=n, freq="D").astype(str)
            rng = np.random.default_rng(29)

            main = pd.DataFrame(
                {
                    "date": dates,
                    "target": rng.normal(size=n),
                    "x_main": rng.normal(size=n),
                }
            )
            p_main = root / "main.csv"
            main.to_csv(p_main, index=False)

            fd, db_tmp = tempfile.mkstemp(suffix=".sqlite3")
            os.close(fd)
            db_path = Path(db_tmp)
            try:
                extra = pd.DataFrame({"date": dates, "wind": rng.normal(loc=3.0, scale=0.7, size=n)})
                with sqlite3.connect(str(db_path)) as conn:
                    extra.to_sql("weather", conn, index=False, if_exists="replace")

                payload = {
                    "data": {
                        "sources": [
                            {"name": "main", "kind": "csv", "path": str(p_main)},
                            {
                                "name": "weather_db",
                                "kind": "sqlite_sql",
                                "db_path": str(db_path),
                                "sql": "SELECT date, wind FROM weather",
                                "prefix": "db_",
                            },
                        ],
                        "merge_on": ["date"],
                        "target_col": "target",
                        "date_col": "date",
                        "feature_recipe": "raw_all_numeric",
                        "split_mode": "ratio",
                        "test_ratio": 0.2,
                        "random_seed": 7,
                    },
                    "train": {
                        "trainer_key": "ridge",
                        "trainer_params": {"l2": 1.0},
                        "run_name": "compose_sqlite_csv_test",
                        "output_dir": str(root / "runs_sqlite_csv"),
                    },
                }

                spec = build_scaffold_spec(payload)
                result = run_project_scaffold(spec)
                self.assertIn("train", result.metrics)
                meta = dict(result.report.get("data", {}).get("metadata", {}))
                self.assertEqual(str(meta.get("source_mode")), "composed")
                self.assertGreaterEqual(len(tuple(meta.get("data_sources", ()))), 2)
            finally:
                gc.collect()
                try:
                    db_path.unlink(missing_ok=True)
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()

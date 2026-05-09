from __future__ import annotations

import gc
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

from catalog import catalog_ui_snapshot, materialize_catalog_db
from catalog.dashboard import (
    _build_deep_link_query,
    _deep_link_with_nav_action,
    _floating_nav_markup,
    _read_query_params,
    build_streamlit_command,
    dashboard_script_path,
)
from catalog import dashboard_shared as dashboard_shared_module
from core.experiment_db import experiment_db_config_info, resolve_experiment_db_target
from core.flow_experiment_tracker import ExperimentTrackerCapability
from core.state.context_keys import RUN_STAGE
from experiment.dashboard import (
    _build_deep_link_query as build_experiment_deep_link_query,
    _read_query_params as read_experiment_query_params,
    build_streamlit_command as build_experiment_streamlit_command,
    dashboard_script_path as experiment_dashboard_script_path,
)


class TestMLBlackCli(unittest.TestCase):
    @staticmethod
    def _prepare_overridden_catalog(db_path: Path) -> str:
        materialize_catalog_db(str(db_path), profile="framework-core")
        override_summary = "cli db override summary"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                """
                UPDATE catalog_entries
                SET summary=?, search_text=?
                WHERE profile=? AND key=?
                """,
                (
                    override_summary,
                    override_summary.lower(),
                    "framework-core",
                    "preset:mlp_torch",
                ),
            )
            conn.commit()
        gc.collect()
        return override_summary

    @staticmethod
    def _prepare_experiment_db(db_path: Path) -> str:
        cap = ExperimentTrackerCapability(
            db_path=str(db_path),
            namespace="ut_cli",
        )
        ctx: dict[str, object] = {
            "run_name": "cli_experiment_surface",
            "context_refs": {RUN_STAGE: "report"},
            "snapshot_count": 2,
            "output_dir": str(db_path.parent / "out"),
            "trainer": SimpleNamespace(name="symbolic_torch_interval"),
            "report": {
                "run_name": "cli_experiment_surface",
                "trainer_name": "symbolic_torch_interval",
                "training": {
                    "requested_init": {"mode": "warm_start"},
                    "task_signature": {
                        "symbolic_family_signature": "sig_cli",
                        "metadata": {
                            "symbolic_family": {
                                "search_family_signature_contracts": [
                                    {"mechanism_key": "beam_selection", "consume": ["gradient_signal"]}
                                ]
                            }
                        },
                    },
                },
                "artifact": {
                    "artifact_id": "artifact_cli_interval",
                    "symbolic_artifact_schema": {
                        "head_semantics": {"task": "interval"},
                        "complexity_metrics": {"term_count": 3},
                        "regime_structure": {
                            "mode": "piecewise",
                            "local_regime_count": 2,
                        },
                        "basis_structure": {
                            "basis_scope": "global+local",
                            "basis_count": 5,
                        },
                        "assembler_structure": {
                            "assembler_mode": "piecewise_budgeted_symbolic_regression",
                            "output_expression_count": 6,
                        },
                        "piecewise_gate_basis": {
                            "status": "enabled",
                        },
                        "truth_contract_recovery": {
                            "status": "reported",
                            "exact_basis_hit_score": 0.75,
                            "exact_term_recovery_score": 0.5,
                        },
                        "orthogonal_search_objective": {
                            "status": "reported",
                            "outer_score": 1.42,
                            "inner_fit_score": 0.84,
                        },
                        "stability_metrics": {
                            "fold_count": 3,
                            "fold_summary": {
                                "rmse_mean": 0.31,
                                "rmse_std": 0.02,
                                "coverage_error_mean": 0.04,
                            },
                            "rmse_mean": 0.31,
                            "rmse_std": 0.02,
                            "coverage_error_mean": 0.04,
                        },
                    },
                },
            },
        }
        cap.on_flow_start(ctx)
        cap.on_flow_finish(ctx)
        tracker = dict(ctx.get("experiment_tracker", {}))
        return str(tracker.get("run_id", ""))

    def test_cli_catalog_summary(self) -> None:
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "mlblack", "catalog", "summary", "--profile", "framework-core"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn('"profile": "framework-core"', proc.stdout)

    def test_cli_catalog_field_filters_and_schema(self) -> None:
        root = Path(__file__).resolve().parents[1]

        list_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlblack",
                "catalog",
                "list",
                "--profile",
                "framework-core",
                "--kind",
                "preset",
                "--field",
                "family=neural",
                "--format",
                "json",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(list_proc.returncode, 0, msg=list_proc.stderr)
        self.assertIn('"key": "preset:mlp_torch"', list_proc.stdout)
        self.assertIn('"key": "preset:sklearn_mlp"', list_proc.stdout)

        schema_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlblack",
                "catalog",
                "schema",
                "--profile",
                "framework-core",
                "--kind",
                "preset",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(schema_proc.returncode, 0, msg=schema_proc.stderr)
        self.assertIn('"family"', schema_proc.stdout)
        self.assertIn('"heads"', schema_proc.stdout)

    def test_cli_catalog_values(self) -> None:
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlblack",
                "catalog",
                "values",
                "family",
                "--profile",
                "framework-core",
                "--kind",
                "preset",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn('"linear"', proc.stdout)
        self.assertIn('"neural"', proc.stdout)
        self.assertIn('"tree_ensemble"', proc.stdout)

    def test_cli_catalog_component_and_plugin_kinds(self) -> None:
        root = Path(__file__).resolve().parents[1]

        component_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlblack",
                "catalog",
                "list",
                "--profile",
                "framework-core",
                "--kind",
                "component",
                "--field",
                "component_surface=runtime_mechanism",
                "--format",
                "json",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(component_proc.returncode, 0, msg=component_proc.stderr)
        self.assertIn('"key": "component:state_signal_view.gradient_norm"', component_proc.stdout)

        plugin_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlblack",
                "catalog",
                "list",
                "--profile",
                "framework-core",
                "--kind",
                "plugin",
                "--field",
                "plugin_surface=flow_plugin",
                "--format",
                "json",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(plugin_proc.returncode, 0, msg=plugin_proc.stderr)
        self.assertIn('"key": "plugin:report_writer"', plugin_proc.stdout)

    def test_cli_catalog_snapshot_and_neighbors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            catalog_db_path = Path(tmp) / "catalog_neighbors.sqlite3"
            materialize_catalog_db(str(catalog_db_path), profile="framework-core")

            snapshot_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "snapshot",
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--field",
                    "family=neural",
                    "--selected",
                    "preset:mlp_torch",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(snapshot_proc.returncode, 0, msg=snapshot_proc.stderr)
            self.assertIn('"selected"', snapshot_proc.stdout)
            self.assertIn('"preset:mlp_torch"', snapshot_proc.stdout)

            neighbors_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "neighbors",
                    "preset:mlp_torch",
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(neighbors_proc.returncode, 0, msg=neighbors_proc.stderr)
            self.assertIn('"family"', neighbors_proc.stdout)
            self.assertIn('"head:point"', neighbors_proc.stdout)

            relation_edges_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "relation-edges",
                    "--db-path",
                    str(catalog_db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--relation-name",
                    "components",
                    "--source-key",
                    "preset:mlp_torch",
                    "--limit",
                    "20",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(relation_edges_proc.returncode, 0, msg=relation_edges_proc.stderr)
            self.assertIn('"relation_name": "components"', relation_edges_proc.stdout)
            self.assertIn('"component:state_signal_view.gradient_norm"', relation_edges_proc.stdout)

            relation_keys_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "relation-keys",
                    "--db-path",
                    str(catalog_db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--relation-name",
                    "components",
                    "--limit",
                    "20",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(relation_keys_proc.returncode, 0, msg=relation_keys_proc.stderr)
            self.assertIn('"relation_name": "components"', relation_keys_proc.stdout)
            self.assertIn('"component:state_signal_view.gradient_norm"', relation_keys_proc.stdout)

    def test_cli_catalog_source_protocol_can_auto_route_to_db(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_auto.sqlite3"
            override_summary = self._prepare_overridden_catalog(db_path)
            env = dict(os.environ)
            env["MLBLACK_CATALOG_DB_URL"] = f"sqlite+pysqlite:///{db_path.as_posix()}"
            env["MLBLACK_CATALOG_DB_MODE"] = "only"

            show_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "show",
                    "preset:mlp_torch",
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(show_proc.returncode, 0, msg=show_proc.stderr)
            self.assertIn(override_summary, show_proc.stdout)

            source_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "source",
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(source_proc.returncode, 0, msg=source_proc.stderr)
            self.assertIn('"effective_source": "db"', source_proc.stdout)
            self.assertIn('"source_mode": "only"', source_proc.stdout)

            target_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "target",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(target_proc.returncode, 0, msg=target_proc.stderr)
            self.assertIn('"enabled": true', target_proc.stdout)
            self.assertIn('"db_backend": "sqlite"', target_proc.stdout)
            gc.collect()

    def test_cli_catalog_db_materialize_respects_readonly_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_readonly.sqlite3"
            config_path = Path(tmp) / "db.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[catalog_db]",
                        "enabled = true",
                        'mode = "prefer"',
                        'backend = "sqlite"',
                        f'path = "{db_path.as_posix()}"',
                        "readonly = true",
                    ]
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["MLBLACK_CATALOG_DB_CONFIG"] = str(config_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "materialize",
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("readonly", proc.stderr.lower())

    def test_cli_catalog_db_materialize_summary_and_show(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog.sqlite3"

            materialize_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "materialize",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(materialize_proc.returncode, 0, msg=materialize_proc.stderr)
            self.assertIn('"profile": "framework-core"', materialize_proc.stdout)
            self.assertIn('"db_backend": "sqlite"', materialize_proc.stdout)

            target_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "target",
                    "--db-path",
                    "postgresql://demo:secret@localhost:5432/mlblack_catalog",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(target_proc.returncode, 0, msg=target_proc.stderr)
            self.assertIn('"db_backend": "postgresql"', target_proc.stdout)
            self.assertIn('"db_driver": "postgresql+psycopg"', target_proc.stdout)
            self.assertNotIn("secret", target_proc.stdout)

            summary_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "summary",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(summary_proc.returncode, 0, msg=summary_proc.stderr)
            self.assertIn('"materialized": true', summary_proc.stdout)

            show_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "show",
                    "preset:mlp_torch",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(show_proc.returncode, 0, msg=show_proc.stderr)
            self.assertIn('"key": "preset:mlp_torch"', show_proc.stdout)
            self.assertIn('"family": "neural"', show_proc.stdout)

            list_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "list",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--field",
                    "family=neural",
                    "--format",
                    "json",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(list_proc.returncode, 0, msg=list_proc.stderr)
            self.assertIn('"key": "preset:mlp_torch"', list_proc.stdout)
            self.assertIn('"key": "preset:sklearn_mlp"', list_proc.stdout)

            search_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "search",
                    "gradient_norm",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--format",
                    "json",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(search_proc.returncode, 0, msg=search_proc.stderr)
            self.assertIn('"key": "preset:mlp_torch"', search_proc.stdout)

            values_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "values",
                    "family",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(values_proc.returncode, 0, msg=values_proc.stderr)
            self.assertIn('"field": "family"', values_proc.stdout)
            self.assertIn('"neural"', values_proc.stdout)

            facets_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "facets",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--field",
                    "family=neural",
                    "--facet-field",
                    "runtime_backend",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(facets_proc.returncode, 0, msg=facets_proc.stderr)
            self.assertIn('"runtime_backend"', facets_proc.stdout)
            self.assertIn('"torch"', facets_proc.stdout)

            neighbors_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "neighbors",
                    "preset:mlp_torch",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(neighbors_proc.returncode, 0, msg=neighbors_proc.stderr)
            self.assertIn('"family"', neighbors_proc.stdout)
            self.assertIn('"components"', neighbors_proc.stdout)

            snapshot_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "catalog",
                    "db",
                    "snapshot",
                    "--db-path",
                    str(db_path),
                    "--profile",
                    "framework-core",
                    "--kind",
                    "preset",
                    "--field",
                    "family=neural",
                    "--selected",
                    "preset:mlp_torch",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(snapshot_proc.returncode, 0, msg=snapshot_proc.stderr)
            self.assertIn('"selected"', snapshot_proc.stdout)
            self.assertIn('"components"', snapshot_proc.stdout)

    def test_cli_catalog_ui_help(self) -> None:
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "mlblack", "catalog", "ui", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("standalone catalog page", proc.stdout.lower())
        self.assertIn("--kind", proc.stdout)
        self.assertIn("--scope", proc.stdout)
        self.assertIn("--project-path", proc.stdout)
        self.assertIn("--db-path", proc.stdout)
        self.assertIn("--source-mode", proc.stdout)
        self.assertIn("sqlalchemy url", proc.stdout.lower())
        self.assertIn("component", proc.stdout)

    def test_catalog_ui_command_builder(self) -> None:
        command = build_streamlit_command(
            profile="framework-core",
            scope="project",
            kind="head",
            query="interval",
            project_path="C:/tmp/demo_project",
            include_global=True,
            db_path="catalog.sqlite3",
            source_mode="only",
            column_mode="full",
            page_size=25,
            results_collapse="collapsed",
            host="127.0.0.1",
            port=8601,
            headless=True,
        )
        self.assertEqual(command[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertIn(str(dashboard_script_path()), command)
        self.assertIn("--server.address", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--server.port", command)
        self.assertIn("8601", command)
        self.assertIn("--server.headless", command)
        self.assertIn("--profile", command)
        self.assertIn("framework-core", command)
        self.assertIn("--scope", command)
        self.assertIn("project", command)
        self.assertIn("--kind", command)
        self.assertIn("head", command)
        self.assertIn("--query", command)
        self.assertIn("interval", command)
        self.assertIn("--project-path", command)
        self.assertIn("C:/tmp/demo_project", command)
        self.assertIn("--include-global", command)
        self.assertIn("--db-path", command)
        self.assertIn("catalog.sqlite3", command)
        self.assertIn("--source-mode", command)
        self.assertIn("only", command)
        self.assertIn("--column-mode", command)
        self.assertIn("full", command)
        self.assertIn("--page-size", command)
        self.assertIn("25", command)
        self.assertIn("--results-collapse", command)
        self.assertIn("collapsed", command)

    def test_catalog_ui_current_selection_surface_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "catalog" / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("当前选中", source)
        self.assertIn("\u4e0a\u4e00\u9879", source)
        self.assertIn("\u4e0b\u4e00\u9879", source)
        self.assertIn("\u5b9a\u4f4d\u5230\u7ed3\u679c\u533a", source)
        self.assertIn("\u663e\u793a\u5b83", source)
        self.assertIn("\u8fd4\u56de\u4e0a\u4e00\u4e2a", source)
        self.assertIn("\u6e05\u9664\u9009\u4e2d", source)
        self.assertIn("\u5f53\u524d\u9009\u4e2d\u9879\u4ecd\u4fdd\u7559\u5728\u53f3\u4fa7\u8be6\u60c5\u91cc", source)

    def test_catalog_ui_deep_link_serializes_field_filters(self) -> None:
        query = _build_deep_link_query(
            profile="framework-core",
            scope="project",
            kind="preset",
            query="mlp",
            selected="preset:mlp_torch",
            project_path="C:/tmp/demo_project",
            include_global=True,
            db_path="postgresql://catalog",
            source_mode="prefer",
            sort_by="title",
            sort_dir="desc",
            detail_tab="relations",
            open_relations="family,head",
            column_mode="full",
            page_size=25,
            results_collapse="collapsed",
            field_filters={"family": "neural", "head": "point"},
        )

        self.assertIn("profile=framework-core", query)
        self.assertIn("scope=project", query)
        self.assertIn("sort_by=title", query)
        self.assertIn("sort_dir=desc", query)
        self.assertIn("detail_tab=relations", query)
        self.assertIn("open_relations=family%2Chead", query)
        self.assertIn("column_mode=full", query)
        self.assertIn("page_size=25", query)
        self.assertIn("results_collapse=collapsed", query)
        self.assertIn("f_family=neural", query)
        self.assertIn("f_head=point", query)
        self.assertIn("selected=preset%3Amlp_torch", query)

    def test_catalog_ui_deep_link_with_nav_action(self) -> None:
        query = _build_deep_link_query(
            profile="framework-core",
            scope="framework",
            kind="preset",
            query="mlp",
            selected="preset:mlp_torch",
            project_path="",
            include_global=False,
            db_path="",
            source_mode="prefer",
            sort_by="title",
            sort_dir="desc",
            detail_tab="relations",
            open_relations="family,head",
            column_mode="full",
            page_size=25,
            results_collapse="collapsed",
            field_filters={"family": "neural"},
        )

        linked = _deep_link_with_nav_action(query, action="locate_selected")
        params = parse_qs(linked.lstrip("?").split("#", 1)[0])

        self.assertEqual(params["nav_action"], ["locate_selected"])
        self.assertEqual(params["selected"], ["preset:mlp_torch"])

    def test_catalog_ui_floating_nav_markup_has_targets(self) -> None:
        active = _floating_nav_markup(
            locate_target="catalog-results-anchor",
            top_target="catalog-page-top",
        )
        disabled = _floating_nav_markup(
            locate_target=None,
            locate_tooltip="???????",
            top_target="catalog-page-top",
        )

        self.assertIn("data-scroll-target='catalog-results-anchor'", active)
        self.assertIn("catalog-page-top", active)
        self.assertIn("data-tooltip='\u5b9a\u4f4d\u5f53\u524d\u9009\u4e2d\u9879'", active)
        self.assertIn("data-tooltip='\u56de\u5230\u9875\u9762\u9876\u90e8'", active)
        self.assertIn("catalog-fab-disabled", disabled)

    def test_catalog_ui_query_reader_restores_field_filters(self) -> None:
        class _FakeStreamlit:
            query_params = {
                "profile": "framework-core",
                "scope": "project",
                "kind": "preset",
                "selected": "preset:mlp_torch",
                "sort_by": "title",
                "sort_dir": "desc",
                "detail_tab": "relations",
                "open_relations": "family,head",
                "column_mode": "full",
                "page_size": "25",
                "results_collapse": "collapsed",
                "f_family": "neural",
                "f_head": "point",
            }

        base_params, field_filters = _read_query_params(_FakeStreamlit())

        self.assertEqual(base_params["profile"], "framework-core")
        self.assertEqual(base_params["selected"], "preset:mlp_torch")
        self.assertEqual(base_params["sort_by"], "title")
        self.assertEqual(base_params["sort_dir"], "desc")
        self.assertEqual(base_params["detail_tab"], "relations")
        self.assertEqual(base_params["open_relations"], "family,head")
        self.assertEqual(base_params["column_mode"], "full")
        self.assertEqual(base_params["page_size"], "25")
        self.assertEqual(base_params["results_collapse"], "collapsed")
        self.assertEqual(field_filters["family"], ("neural",))
        self.assertEqual(field_filters["head"], ("point",))

    def test_sync_query_filters_keeps_live_session_value_when_url_is_empty(self) -> None:
        class _FakeStreamlit:
            def __init__(self) -> None:
                self.session_state = {
                    dashboard_shared_module.facet_state_key("framework", "preset", "family"): "neural",
                }

        fake = _FakeStreamlit()
        dashboard_shared_module.sync_query_filters_to_session(
            fake,
            scope="framework",
            kind="preset",
            facet_fields=("family", "head"),
            query_filters={},
            multi_value=False,
        )
        self.assertEqual(
            fake.session_state[dashboard_shared_module.facet_state_key("framework", "preset", "family")],
            "neural",
        )
        self.assertEqual(
            fake.session_state[dashboard_shared_module.facet_state_key("framework", "preset", "head")],
            "",
        )

    def test_catalog_ui_deep_link_roundtrip_restores_result_layout_and_snapshot(self) -> None:
        original_query = _build_deep_link_query(
            profile="framework-core",
            scope="framework",
            kind="preset",
            query="mlp",
            selected="preset:mlp_torch",
            project_path="",
            include_global=False,
            db_path="",
            source_mode="prefer",
            sort_by="title",
            sort_dir="desc",
            detail_tab="relations",
            open_relations="family,head",
            column_mode="full",
            page_size=100,
            results_collapse="collapsed",
            field_filters={"family": "neural", "head": "point"},
        )

        class _FakeStreamlit:
            query_params = {key: values[-1] for key, values in parse_qs(original_query.lstrip("?")).items()}

        base_params, field_filters = _read_query_params(_FakeStreamlit())
        rebuilt_query = _build_deep_link_query(
            profile=base_params["profile"],
            scope=base_params["scope"],
            kind=base_params["kind"],
            query=base_params["query"],
            selected=base_params["selected"],
            project_path=base_params.get("project_path", ""),
            include_global=base_params.get("include_global", "") == "1",
            db_path=base_params.get("db_path", ""),
            source_mode=base_params.get("source_mode", ""),
            sort_by=base_params["sort_by"],
            sort_dir=base_params["sort_dir"],
            detail_tab=base_params["detail_tab"],
            open_relations=base_params["open_relations"],
            column_mode=base_params["column_mode"],
            page_size=int(base_params["page_size"]),
            results_collapse=base_params["results_collapse"],
            field_filters=field_filters,
        )
        snapshot = catalog_ui_snapshot(
            profile=base_params["profile"],
            scope=base_params["scope"],
            kind=base_params["kind"],
            query=base_params["query"],
            field_filters=field_filters,
            selected_key=base_params["selected"],
            db_path=base_params.get("db_path", "") or None,
            source_mode=base_params.get("source_mode", "") or None,
        )

        self.assertEqual(parse_qs(rebuilt_query.lstrip("?")), parse_qs(original_query.lstrip("?")))
        self.assertIsNotNone(snapshot["selected"])
        self.assertEqual(snapshot["selected"]["key"], "preset:mlp_torch")
        self.assertTrue(any(item["key"] == "preset:mlp_torch" for item in snapshot["items"]))

    def test_experiment_dashboard_command_builder(self) -> None:
        command = build_experiment_streamlit_command(
            db_path="runs/experiments.sqlite3",
            limit=200,
            host="127.0.0.1",
            port=8602,
            headless=True,
        )
        self.assertEqual(command[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertIn(str(experiment_dashboard_script_path()), command)
        self.assertIn("--server.address", command)
        self.assertIn("127.0.0.1", command)
        self.assertIn("--server.port", command)
        self.assertIn("8602", command)
        self.assertIn("--server.headless", command)
        self.assertIn("--db", command)
        self.assertIn("runs/experiments.sqlite3", command)
        self.assertIn("--limit", command)
        self.assertIn("200", command)

    def test_experiment_dashboard_query_reader_restores_filters_and_selection(self) -> None:
        class _FakeStreamlit:
            query_params = {
                "db": "runs/experiments.sqlite3",
                "limit": "180",
                "view": "artifact_catalog",
                "selected": "artifact:run_demo:artifact_interval",
                "f_artifact_trainer_name": "symbolic_torch_interval",
                "f_artifact_head_task": "interval",
                "f_artifact_regime_mode": "piecewise",
                "f_artifact_basis_scope": "global+local",
                "f_artifact_assembler_mode": "piecewise_budgeted_symbolic_regression",
                "f_artifact_piecewise_gate_status": "enabled",
                "f_artifact_fold_summary": "present",
                "f_artifact_rmse_std_lte": "0.03",
            }

        base_params, field_filters = read_experiment_query_params(_FakeStreamlit())

        self.assertEqual(base_params["db"], "runs/experiments.sqlite3")
        self.assertEqual(base_params["limit"], "180")
        self.assertEqual(base_params["view"], "artifact_catalog")
        self.assertEqual(base_params["selected"], "artifact:run_demo:artifact_interval")
        self.assertEqual(field_filters["artifact_trainer_name"], ("symbolic_torch_interval",))
        self.assertEqual(field_filters["artifact_head_task"], ("interval",))
        self.assertEqual(field_filters["artifact_regime_mode"], ("piecewise",))
        self.assertEqual(field_filters["artifact_basis_scope"], ("global+local",))
        self.assertEqual(field_filters["artifact_assembler_mode"], ("piecewise_budgeted_symbolic_regression",))
        self.assertEqual(field_filters["artifact_piecewise_gate_status"], ("enabled",))
        self.assertEqual(field_filters["artifact_fold_summary"], ("present",))
        self.assertEqual(field_filters["artifact_rmse_std_lte"], ("0.03",))

    def test_experiment_dashboard_deep_link_roundtrip(self) -> None:
        original_query = build_experiment_deep_link_query(
            base_params={
                "db": "runs/experiments.sqlite3",
                "limit": "220",
                "view": "run_catalog",
                "selected": "run:run_demo",
            },
            field_filters={
                "run_status": "completed",
                "run_trainer_name": "symbolic_torch_interval",
                "run_fold_summary": "present",
                "run_surface_key": "flow:demo",
                "run_family_ref": "family:symbolic",
                "run_assembly_signature": "sig_demo",
                "run_regime_mode": "piecewise",
                "run_basis_scope": "global+local",
                "run_assembler_mode": "piecewise_budgeted_symbolic_regression",
                "run_piecewise_gate_status": "enabled",
                "run_rmse_std_lte": "0.03",
            },
        )

        class _FakeStreamlit:
            query_params = {key: values[-1] for key, values in parse_qs(original_query.lstrip("?")).items()}

        base_params, field_filters = read_experiment_query_params(_FakeStreamlit())
        rebuilt_query = build_experiment_deep_link_query(
            base_params=base_params,
            field_filters=field_filters,
        )

        self.assertEqual(parse_qs(rebuilt_query.lstrip("?")), parse_qs(original_query.lstrip("?")))

    def test_experiment_dashboard_uses_top_product_skeleton(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "experiment" / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn("experiment-hero", source)
        self.assertIn("experiment-inline-filters", source)
        self.assertIn("\u5b9e\u9a8c\u7ed3\u679c / \u4ea7\u7269\u53cc\u89c6\u56fe\u5de5\u4f5c\u53f0", source)
        self.assertNotIn("st.sidebar", source)

    def test_experiment_dashboard_uses_clickable_results_table(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "experiment" / "dashboard.py").read_text(encoding="utf-8")
        self.assertIn('on_select="rerun"', source)
        self.assertIn('selection_mode="single-row"', source)
        self.assertIn("当前选中", source)

    def test_experiment_db_config_resolver_reads_toml_protocol(self) -> None:
        old_config = os.environ.get("MLBLACK_EXPERIMENT_DB_CONFIG")
        old_url = os.environ.get("MLBLACK_EXPERIMENT_DB_URL")
        old_mode = os.environ.get("MLBLACK_EXPERIMENT_DB_MODE")
        old_readonly = os.environ.get("MLBLACK_EXPERIMENT_DB_READONLY")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / "experiment_db.toml"
                config_path.write_text(
                    "\n".join(
                        [
                            "[experiment_db]",
                            "enabled = true",
                            'mode = "prefer"',
                            "readonly = false",
                            'backend = "postgresql"',
                            'host = "127.0.0.1"',
                            "port = 5432",
                            'user = "postgres"',
                            'password = "secret_pw"',
                            'database = "mlblack_runtime"',
                            'driver = "postgresql+psycopg"',
                        ]
                    ),
                    encoding="utf-8",
                )
                os.environ["MLBLACK_EXPERIMENT_DB_CONFIG"] = str(config_path)
                os.environ.pop("MLBLACK_EXPERIMENT_DB_URL", None)
                os.environ.pop("MLBLACK_EXPERIMENT_DB_MODE", None)
                os.environ.pop("MLBLACK_EXPERIMENT_DB_READONLY", None)

                target = resolve_experiment_db_target()
                info = experiment_db_config_info()

                self.assertEqual(info["source"], "file")
                self.assertEqual(info["db_backend"], "postgresql")
                self.assertIn("mlblack_runtime", target)
                self.assertTrue(target.startswith("postgresql+psycopg://postgres:secret_pw@127.0.0.1:5432/"))
        finally:
            if old_config is None:
                os.environ.pop("MLBLACK_EXPERIMENT_DB_CONFIG", None)
            else:
                os.environ["MLBLACK_EXPERIMENT_DB_CONFIG"] = old_config
            if old_url is None:
                os.environ.pop("MLBLACK_EXPERIMENT_DB_URL", None)
            else:
                os.environ["MLBLACK_EXPERIMENT_DB_URL"] = old_url
            if old_mode is None:
                os.environ.pop("MLBLACK_EXPERIMENT_DB_MODE", None)
            else:
                os.environ["MLBLACK_EXPERIMENT_DB_MODE"] = old_mode
            if old_readonly is None:
                os.environ.pop("MLBLACK_EXPERIMENT_DB_READONLY", None)
            else:
                os.environ["MLBLACK_EXPERIMENT_DB_READONLY"] = old_readonly


    def test_cli_scaffold_init(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "mlblack_cli_init_case"
            proc = subprocess.run(
                [sys.executable, "-m", "mlblack", "scaffold", "init", "--path", str(target)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue((target / "run_train.py").exists())
            self.assertTrue((target / "configs" / "train_config.json").exists())
            self.assertTrue((target / "catalog" / "entries.toml").exists())

    def test_cli_experiment_surface_summary_and_queries(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "experiments.sqlite3"
            run_id = self._prepare_experiment_db(db_path)
            self.assertTrue(run_id)

            summary_proc = subprocess.run(
                [sys.executable, "-m", "mlblack", "experiment", "summary", "--db", str(db_path)],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(summary_proc.returncode, 0, msg=summary_proc.stderr)
            self.assertIn('"experiment_run_catalog": 1', summary_proc.stdout)
            self.assertIn('"experiment_artifact_catalog": 1', summary_proc.stdout)

            summary_default_proc = subprocess.run(
                [sys.executable, "-m", "mlblack", "experiment", "summary"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "MLBLACK_EXPERIMENT_DB_URL": str(db_path)},
            )
            self.assertEqual(summary_default_proc.returncode, 0, msg=summary_default_proc.stderr)
            self.assertIn('"experiment_run_catalog": 1', summary_default_proc.stdout)

            list_runs_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "experiment",
                    "list-runs",
                    "--db",
                    str(db_path),
                    "--has-fold-summary",
                    "--max-rmse-std",
                    "0.03",
                    "--min-exact-basis-hit-score",
                    "0.7",
                    "--min-outer-objective-score",
                    "1.0",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(list_runs_proc.returncode, 0, msg=list_runs_proc.stderr)
            self.assertIn('"artifact_id": "artifact_cli_interval"', list_runs_proc.stdout)
            self.assertIn('"exact_basis_hit_score": 0.75', list_runs_proc.stdout)
            self.assertIn('"surface_key": "flow:cli_experiment_surface"', list_runs_proc.stdout)
            self.assertIn('"family_ref": "family:symbolic"', list_runs_proc.stdout)
            self.assertIn('"regime_mode": "piecewise"', list_runs_proc.stdout)
            self.assertIn('"basis_scope": "global+local"', list_runs_proc.stdout)

            filtered_runs_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "experiment",
                    "list-runs",
                    "--db",
                    str(db_path),
                    "--surface-key",
                    "flow:cli_experiment_surface",
                    "--family-ref",
                    "family:symbolic",
                    "--regime-mode",
                    "piecewise",
                    "--basis-scope",
                    "global+local",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(filtered_runs_proc.returncode, 0, msg=filtered_runs_proc.stderr)
            self.assertIn('"run_id":', filtered_runs_proc.stdout)
            self.assertIn('"assembly_signature":', filtered_runs_proc.stdout)

            show_run_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "experiment",
                    "show-run",
                    "--db",
                    str(db_path),
                    "--run-id",
                    run_id,
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(show_run_proc.returncode, 0, msg=show_run_proc.stderr)
            self.assertIn('"run_id":', show_run_proc.stdout)
            self.assertIn('"coverage_error_mean": 0.04', show_run_proc.stdout)
            self.assertIn('"assembler_mode": "piecewise_budgeted_symbolic_regression"', show_run_proc.stdout)

            show_artifact_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlblack",
                    "experiment",
                    "show-artifact",
                    "--db",
                    str(db_path),
                    "--run-id",
                    run_id,
                    "--artifact-id",
                    "artifact_cli_interval",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(show_artifact_proc.returncode, 0, msg=show_artifact_proc.stderr)
            self.assertIn('"head_task": "interval"', show_artifact_proc.stdout)
            self.assertIn('"piecewise_gate_status": "enabled"', show_artifact_proc.stdout)


if __name__ == "__main__":
    unittest.main()





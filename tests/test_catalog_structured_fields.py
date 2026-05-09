from __future__ import annotations

import gc
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from catalog import (
    catalog_db_config_info,
    catalog_db_facets,
    catalog_db_field_values,
    catalog_db_list_entries,
    catalog_db_neighbors,
    catalog_db_relation_edges,
    catalog_db_relation_keys,
    catalog_db_search_entries,
    catalog_db_show_entry,
    catalog_db_summary,
    catalog_db_target_info,
    catalog_db_ui_snapshot,
    catalog_facets,
    catalog_neighbors,
    catalog_schema,
    catalog_source_info,
    catalog_ui_snapshot,
    field_values,
    list_entries,
    materialize_catalog_db,
    materialize_catalog_sqlite,
    search_entries,
    show_entry,
)
from catalog.sql_store import _CATALOG_SCALARS


class TestCatalogStructuredFields(unittest.TestCase):
    @staticmethod
    def _materialize_overridden_catalog(db_path: Path, *, profile: str = "framework-core") -> str:
        materialize_catalog_db(str(db_path), profile=profile)
        override_summary = "db override summary for mlp_torch"
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
                    str(profile),
                    "preset:mlp_torch",
                ),
            )
            conn.commit()
        return override_summary

    def test_catalog_can_materialize_to_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog.sqlite3"
            payload = materialize_catalog_db(str(db_path), profile="framework-core")

            self.assertEqual(str(payload.get("profile")), "framework-core")
            self.assertTrue(db_path.exists())
            self.assertGreater(int(payload.get("total", 0)), 0)
            self.assertEqual(str(payload.get("db_backend")), "sqlite")
            self.assertGreater(int(payload.get("relation_edges", 0)), 0)
            self.assertGreater(int(payload.get("relation_keys", 0)), 0)

            summary = catalog_db_summary(str(db_path), profile="framework-core")
            self.assertTrue(bool(summary.get("materialized")))
            self.assertEqual(int(summary.get("total", 0)), int(payload.get("total", 0)))

            item = catalog_db_show_entry(str(db_path), "preset:mlp_torch", profile="framework-core")
            self.assertIsNotNone(item)
            assert item is not None
            self.assertEqual(item.kind, "preset")
            self.assertEqual(str(item.fields.get("family")), "neural")
            self.assertEqual(str(item.fields.get("title_zh")), "Torch 多层感知机")
            self.assertIn("神经网络家族", str(item.fields.get("summary_zh", "")))
            self.assertTrue(tuple(item.fields.get("use_when_zh", ())))
            self.assertIn("component:state_signal_view.gradient_norm", tuple(item.relations.get("components", ())))
            self.assertIn("plugin:experiment_tracker", tuple(item.relations.get("plugins", ())))
            symbolic_preset = catalog_db_show_entry(str(db_path), "preset:symbolic", profile="framework-core")
            self.assertIsNotNone(symbolic_preset)
            assert symbolic_preset is not None
            self.assertEqual(str(symbolic_preset.fields.get("artifact_schema_key")), "symbolic_artifact_v1")
            self.assertIn("complexity_metrics", tuple(symbolic_preset.fields.get("artifact_schema_fields", ())))
            self.assertIn("fold_summary", tuple(symbolic_preset.fields.get("artifact_stability_fields", ())))
            self.assertIn("beam_selection", tuple(symbolic_preset.fields.get("search_mechanism_keys", ())))
            self.assertIn("symbolic_torch", tuple(symbolic_preset.fields.get("symbolic_route_keys", ())))
            self.assertIn("torch", tuple(symbolic_preset.fields.get("symbolic_route_backends", ())))

            db_items = catalog_db_list_entries(
                str(db_path),
                profile="framework-core",
                kind="preset",
                field_filters={"family": "neural"},
            )
            db_keys = {entry.key for entry in db_items}
            self.assertEqual(db_keys, {"preset:neural", "preset:mlp_torch", "preset:sklearn_mlp"})

            db_search = catalog_db_search_entries(
                str(db_path),
                "gradient_norm",
                profile="framework-core",
                kind="preset",
                limit=20,
            )
            self.assertIn("preset:mlp_torch", {entry.key for entry in db_search})

            db_search_zh = catalog_db_search_entries(
                str(db_path),
                "梯度范数",
                profile="framework-core",
                kind="component",
                limit=20,
            )
            self.assertIn("component:state_signal_view.gradient_norm", {entry.key for entry in db_search_zh})

            db_values = catalog_db_field_values(
                str(db_path),
                "family",
                profile="framework-core",
                kind="preset",
            )
            self.assertIn("linear", db_values)
            self.assertIn("neural", db_values)
            self.assertIn("tree_ensemble", db_values)
            symbolic_mechanism_values = catalog_db_field_values(
                str(db_path),
                "search_mechanism_keys",
                profile="framework-core",
                kind="preset",
            )
            self.assertIn("beam_selection", symbolic_mechanism_values)
            symbolic_route_values = catalog_db_field_values(
                str(db_path),
                "symbolic_route_keys",
                profile="framework-core",
                kind="preset",
            )
            self.assertIn("symbolic_torch_interval", symbolic_route_values)
            stability_values = catalog_db_field_values(
                str(db_path),
                "artifact_stability_fields",
                profile="framework-core",
                kind="preset",
            )
            self.assertIn("fold_summary", stability_values)

            db_facets = catalog_db_facets(
                str(db_path),
                profile="framework-core",
                kind="preset",
                field_filters={"family": "neural"},
                fields=("runtime_backend", "status"),
            )
            runtime_rows = list(dict(db_facets.get("facets", {})).get("runtime_backend", []))
            runtime_values = {str(row.get("value")) for row in runtime_rows}
            self.assertIn("torch", runtime_values)
            self.assertIn("scikit-learn", runtime_values)
            symbolic_facets = catalog_db_facets(
                str(db_path),
                profile="framework-core",
                kind="preset",
                field_filters={"family": "symbolic"},
                fields=("search_mechanism_keys", "artifact_stability_fields", "symbolic_route_keys"),
            )
            mechanism_rows = list(dict(symbolic_facets.get("facets", {})).get("search_mechanism_keys", []))
            stability_rows = list(dict(symbolic_facets.get("facets", {})).get("artifact_stability_fields", []))
            route_rows = list(dict(symbolic_facets.get("facets", {})).get("symbolic_route_keys", []))
            self.assertIn("beam_selection", {str(row.get("value")) for row in mechanism_rows})
            self.assertIn("fold_summary", {str(row.get("value")) for row in stability_rows})
            self.assertIn("symbolic_stagewise", {str(row.get("value")) for row in route_rows})

            db_neighbors = catalog_db_neighbors(
                str(db_path),
                "preset:mlp_torch",
                profile="framework-core",
            )
            self.assertIn("family", dict(db_neighbors.get("neighbors", {})))
            self.assertIn("components", dict(db_neighbors.get("neighbors", {})))
            self.assertTrue(list(db_neighbors.get("relation_keys", [])))

            relation_edges = catalog_db_relation_edges(
                str(db_path),
                profile="framework-core",
                kind="preset",
                relation_name="components",
                source_key="preset:mlp_torch",
                limit=20,
            )
            self.assertTrue(relation_edges)
            self.assertIn("component:state_signal_view.gradient_norm", {str(row.get("target_key")) for row in relation_edges})

            relation_keys = catalog_db_relation_keys(
                str(db_path),
                profile="framework-core",
                kind="preset",
                relation_name="components",
                limit=20,
            )
            self.assertTrue(relation_keys)
            self.assertIn(
                "component:state_signal_view.gradient_norm",
                {str(row.get("relation_value")) for row in relation_keys},
            )

            db_snapshot = catalog_db_ui_snapshot(
                str(db_path),
                profile="framework-core",
                kind="preset",
                field_filters={"family": "neural"},
                selected_key="preset:mlp_torch",
            )
            self.assertEqual(str((db_snapshot.get("selected") or {}).get("key")), "preset:mlp_torch")
            self.assertIn("components", dict((db_snapshot.get("neighbors") or {}).get("neighbors", {})))
            self.assertTrue(list(((db_snapshot.get("neighbors") or {}).get("relation_keys") or [])))

    def test_catalog_can_materialize_via_sqlalchemy_sqlite_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_url.sqlite3"
            db_url = f"sqlite+pysqlite:///{db_path.as_posix()}"

            payload = materialize_catalog_db(db_url, profile="framework-core")
            self.assertEqual(str(payload.get("db_backend")), "sqlite")
            self.assertTrue(db_path.exists())

            item = catalog_db_show_entry(db_url, "preset:mlp_torch", profile="framework-core")
            self.assertIsNotNone(item)
            assert item is not None
            self.assertEqual(item.kind, "preset")
            self.assertEqual(str(item.fields.get("family")), "neural")

    def test_family_entries_are_materialized(self) -> None:
        items = list_entries(profile="framework-core", kind="family")
        keys = {entry.key for entry in items}

        self.assertIn("family:linear", keys)
        self.assertIn("family:neural", keys)
        self.assertIn("family:tree_ensemble", keys)
        self.assertIn("family:tree_boosting", keys)
        self.assertIn("family:symbolic", keys)

    def test_preset_entry_exposes_structured_family_and_head_fields(self) -> None:
        item = show_entry("preset:mlp_torch", profile="framework-core")
        self.assertIsNotNone(item)
        assert item is not None

        self.assertEqual(item.kind, "preset")
        self.assertEqual(item.fields.get("family"), "neural")
        self.assertEqual(item.fields.get("head"), "point")
        self.assertEqual(tuple(item.fields.get("heads", ())), ("point",))
        self.assertEqual(str(item.fields.get("runtime_backend")), "torch")
        self.assertIn("mlp_torch", tuple(item.fields.get("family_route_keys", ())))
        self.assertEqual(str(item.fields.get("title_zh")), "Torch 多层感知机")
        self.assertIn("神经网络家族", str(item.fields.get("summary_zh", "")))
        self.assertTrue(tuple(item.fields.get("use_when_zh", ())))
        self.assertTrue(bool(item.fields.get("supports_resume")))
        self.assertGreater(int(item.fields.get("component_count", 0)), 0)
        self.assertGreater(int(item.fields.get("provider_count", 0)), 0)
        self.assertGreater(int(item.fields.get("plugin_count", 0)), 0)
        self.assertIn("components", dict(item.relations))
        self.assertIn("providers", dict(item.relations))
        self.assertIn("plugins", dict(item.relations))
        self.assertIn("component:state_signal_view.gradient_norm", tuple(item.relations.get("components", ())))
        self.assertIn("provider:decision_evaluation_bridge", tuple(item.relations.get("providers", ())))
        self.assertIn("plugin:experiment_tracker", tuple(item.relations.get("plugins", ())))

    def test_field_filter_can_slice_presets_by_family(self) -> None:
        items = list_entries(
            profile="framework-core",
            kind="preset",
            field_filters={"family": "neural"},
        )
        keys = {entry.key for entry in items}

        self.assertEqual(keys, {"preset:neural", "preset:mlp_torch", "preset:sklearn_mlp"})

    def test_field_filter_can_find_interval_capable_presets(self) -> None:
        items = list_entries(
            profile="framework-core",
            kind="preset",
            field_filters={"head": "interval"},
        )
        keys = {entry.key for entry in items}

        self.assertIn("preset:symbolic", keys)
        self.assertIn("preset:symbolic_torch_interval", keys)

    def test_head_entries_link_back_to_families_and_presets(self) -> None:
        item = show_entry("head:interval", profile="framework-core")
        self.assertIsNotNone(item)
        assert item is not None

        self.assertEqual(item.kind, "head")
        self.assertIn("symbolic", tuple(item.fields.get("families", ())))
        self.assertIn("symbolic_torch_interval", tuple(item.fields.get("presets", ())))

    def test_symbolic_catalog_entries_surface_artifact_schema_contract(self) -> None:
        family = show_entry("family:symbolic", profile="framework-core")
        self.assertIsNotNone(family)
        assert family is not None
        self.assertEqual(str(family.fields.get("artifact_schema_key")), "symbolic_artifact_v1")
        self.assertIn("complexity_metrics", tuple(family.fields.get("artifact_schema_fields", ())))
        self.assertIn("operator_cost", tuple(family.fields.get("artifact_complexity_fields", ())))

        preset = show_entry("preset:symbolic", profile="framework-core")
        self.assertIsNotNone(preset)
        assert preset is not None
        self.assertEqual(str(preset.fields.get("artifact_schema_key")), "symbolic_artifact_v1")
        self.assertIn("term_contributions", tuple(preset.fields.get("artifact_explainability_fields", ())))
        self.assertTrue(bool(preset.fields.get("artifact_supports_piecewise")))
        self.assertIn("beam_selection", tuple(preset.fields.get("search_mechanism_keys", ())))
        self.assertIn("path_memory", tuple(preset.fields.get("search_replayable_mechanisms", ())))
        self.assertIn("fold_summary", tuple(preset.fields.get("artifact_stability_fields", ())))
        self.assertIn("symbolic_torch", tuple(preset.fields.get("symbolic_route_keys", ())))
        self.assertIn("point", tuple(preset.fields.get("symbolic_route_tasks", ())))

        head = show_entry("head:interval", profile="framework-core")
        self.assertIsNotNone(head)
        assert head is not None
        self.assertEqual(str(head.fields.get("artifact_schema_key")), "symbolic_artifact_v1")
        self.assertIn("residual_std_mean", tuple(head.fields.get("artifact_stability_fields", ())))
        self.assertIn("candidate_budget_policy", tuple(head.fields.get("search_family_signature_mechanisms", ())))
        self.assertIn("symbolic_torch_interval", tuple(head.fields.get("symbolic_route_keys", ())))

        self.assertIn("gradient_projection_guidance", tuple(family.fields.get("search_family_signature_mechanisms", ())))
        self.assertIn("expression_graph_cache", tuple(family.fields.get("search_checkpointable_mechanisms", ())))
        self.assertIn("symbolic_stagewise", tuple(family.fields.get("symbolic_route_keys", ())))
        self.assertIn("ridge", tuple(family.fields.get("symbolic_route_backends", ())))

    def test_symbolic_preset_surface_contract_marks_formal_and_deprecated_entries(self) -> None:
        canonical = show_entry("preset:symbolic", profile="framework-core")
        self.assertIsNotNone(canonical)
        assert canonical is not None
        self.assertEqual(str(canonical.fields.get("surface_status")), "formal")
        self.assertIn("preset:symbolic_torch", tuple(canonical.relations.get("legacy_facades", ())))
        self.assertIn("preset:symbolic_stagewise", tuple(canonical.relations.get("legacy_facades", ())))
        self.assertIn("symbolic_torch", tuple(canonical.fields.get("family_route_keys", ())))

        neural = show_entry("preset:neural", profile="framework-core")
        self.assertIsNotNone(neural)
        assert neural is not None
        self.assertEqual(str(neural.fields.get("surface_status")), "formal")
        self.assertIn("mlp_torch", tuple(neural.fields.get("family_route_keys", ())))
        self.assertIn("sklearn_mlp", tuple(neural.fields.get("family_route_keys", ())))
        self.assertIn("preset:mlp_torch", tuple(neural.relations.get("router_targets", ())))
        self.assertIn("component:state_signal_view.gradient_norm", tuple(neural.relations.get("components", ())))

        linear = show_entry("preset:linear", profile="framework-core")
        self.assertIsNotNone(linear)
        assert linear is not None
        self.assertEqual(str(linear.fields.get("surface_status")), "formal")
        self.assertIn("ridge", tuple(linear.fields.get("family_route_keys", ())))

        tree_boosting = show_entry("preset:tree_boosting", profile="framework-core")
        self.assertIsNotNone(tree_boosting)
        assert tree_boosting is not None
        self.assertEqual(str(tree_boosting.fields.get("surface_status")), "formal")
        self.assertIn("xgboost", tuple(tree_boosting.fields.get("family_route_keys", ())))

        tree_ensemble = show_entry("preset:tree_ensemble", profile="framework-core")
        self.assertIsNotNone(tree_ensemble)
        assert tree_ensemble is not None
        self.assertEqual(str(tree_ensemble.fields.get("surface_status")), "formal")
        self.assertIn("random_forest", tuple(tree_ensemble.fields.get("family_route_keys", ())))
        self.assertIn("adaboost", tuple(tree_ensemble.fields.get("family_route_keys", ())))
        self.assertIn("preset:bagging", tuple(tree_ensemble.relations.get("router_targets", ())))

        legacy = show_entry("preset:symbolic_torch_interval", profile="framework-core")
        self.assertIsNotNone(legacy)
        assert legacy is not None
        self.assertEqual(str(legacy.fields.get("surface_status")), "deprecated")
        self.assertTrue(bool(legacy.fields.get("deprecated_surface")))
        self.assertEqual(str(legacy.fields.get("canonical_preset")), "symbolic")
        self.assertEqual(str(legacy.fields.get("migration_target")), "symbolic")
        self.assertIn("preset:symbolic", tuple(legacy.relations.get("migration_target", ())))

    def test_field_values_and_schema_expose_ui_facet_data(self) -> None:
        families = field_values("family", profile="framework-core", kind="preset")
        self.assertIn("linear", families)
        self.assertIn("neural", families)
        self.assertIn("tree_ensemble", families)
        self.assertIn("tree_boosting", families)

        schema = catalog_schema(profile="framework-core", kind="preset")
        fields = set(schema.get("fields", []))
        self.assertIn("family", fields)
        self.assertIn("head", fields)
        self.assertIn("heads", fields)
        self.assertIn("symbolic_route_keys", fields)
        self.assertIn("title_zh", fields)
        self.assertIn("summary_zh", fields)
        self.assertIn("use_when_zh", fields)

    def test_catalog_search_supports_chinese_bilingual_fields(self) -> None:
        items = search_entries(
            "梯度范数",
            profile="framework-core",
            kind="component",
            limit=20,
        )
        self.assertIn("component:state_signal_view.gradient_norm", {entry.key for entry in items})

        preset_items = search_entries(
            "多层感知机",
            profile="framework-core",
            kind="preset",
            limit=20,
        )
        preset_keys = {entry.key for entry in preset_items}
        self.assertIn("preset:mlp_torch", preset_keys)
        self.assertIn("preset:sklearn_mlp", preset_keys)

    def test_neighbors_and_facets_are_ui_ready(self) -> None:
        neighbors = catalog_neighbors("preset:mlp_torch", profile="framework-core")
        relation_names = set(dict(neighbors.get("neighbors", {})).keys())
        self.assertIn("family", relation_names)
        self.assertIn("heads", relation_names)
        self.assertIn("components", relation_names)
        self.assertIn("providers", relation_names)
        self.assertIn("plugins", relation_names)

        facets = catalog_facets(
            profile="framework-core",
            kind="preset",
            field_filters={"family": "neural"},
            fields=("runtime_backend", "status"),
        )
        runtime_rows = list(dict(facets.get("facets", {})).get("runtime_backend", []))
        runtime_values = {str(row.get("value")) for row in runtime_rows}
        self.assertIn("torch", runtime_values)
        self.assertIn("scikit-learn", runtime_values)

        symbolic_facets = catalog_facets(
            profile="framework-core",
            kind="preset",
            field_filters={"family": "symbolic"},
            fields=("search_mechanism_keys", "artifact_stability_fields", "symbolic_route_keys"),
        )
        self.assertIn(
            "beam_selection",
            {str(row.get("value")) for row in dict(symbolic_facets.get("facets", {})).get("search_mechanism_keys", [])},
        )
        self.assertIn(
            "fold_summary",
            {str(row.get("value")) for row in dict(symbolic_facets.get("facets", {})).get("artifact_stability_fields", [])},
        )
        self.assertIn(
            "symbolic_torch",
            {str(row.get("value")) for row in dict(symbolic_facets.get("facets", {})).get("symbolic_route_keys", [])},
        )

    def test_ui_snapshot_contains_items_facets_and_selected_neighbors(self) -> None:
        payload = catalog_ui_snapshot(
            profile="framework-core",
            kind="preset",
            field_filters={"family": "neural"},
            selected_key="preset:mlp_torch",
        )
        item_keys = {str(item.get("key")) for item in payload.get("items", [])}
        self.assertIn("preset:mlp_torch", item_keys)
        self.assertIn("preset:sklearn_mlp", item_keys)
        self.assertEqual(str((payload.get("selected") or {}).get("key")), "preset:mlp_torch")
        self.assertIn("family", dict((payload.get("neighbors") or {}).get("neighbors", {})))

        symbolic_payload = catalog_ui_snapshot(
            profile="framework-core",
            kind="preset",
            field_filters={"family": "symbolic"},
            selected_key="preset:symbolic",
        )
        mechanism_rows = list(dict((symbolic_payload.get("facets") or {}).get("facets", {})).get("search_mechanism_keys", []))
        stability_rows = list(dict((symbolic_payload.get("facets") or {}).get("facets", {})).get("artifact_stability_fields", []))
        route_rows = list(dict((symbolic_payload.get("facets") or {}).get("facets", {})).get("symbolic_route_keys", []))
        self.assertIn("beam_selection", {str(row.get("value")) for row in mechanism_rows})
        self.assertIn("fold_summary", {str(row.get("value")) for row in stability_rows})
        self.assertIn("symbolic_torch_interval", {str(row.get("value")) for row in route_rows})

    def test_component_entries_cover_runtime_mechanisms_and_biases(self) -> None:
        items = list_entries(profile="framework-core", kind="component")
        keys = {entry.key for entry in items}

        self.assertIn("component:state_signal_view.gradient_norm", keys)
        self.assertIn("component:sample_weighting.loss_adaptive", keys)
        self.assertIn("component:bias.l2_scale", keys)

        gradient_norm = show_entry("component:state_signal_view.gradient_norm", profile="framework-core")
        self.assertIsNotNone(gradient_norm)
        assert gradient_norm is not None
        self.assertEqual(str(gradient_norm.fields.get("component_surface")), "runtime_mechanism")
        self.assertEqual(str(gradient_norm.fields.get("component_kind")), "state_signal_view")
        self.assertIn("neural", tuple(gradient_norm.fields.get("applicable_families", ())))
        self.assertEqual(str(gradient_norm.fields.get("mount_point")), "runtime_mechanism_stack")
        self.assertIn("gradient_norm", tuple(gradient_norm.fields.get("contract_consumes", ())))
        self.assertIn("gradient_norm_ref", tuple(gradient_norm.fields.get("contract_provides", ())))

        l2_scale = show_entry("component:bias.l2_scale", profile="framework-core")
        self.assertIsNotNone(l2_scale)
        assert l2_scale is not None
        self.assertEqual(str(l2_scale.fields.get("component_surface")), "bias")
        self.assertEqual(str(l2_scale.fields.get("mount_point")), "bias_stack")
        self.assertIn("legacy_bias_entry", dict(l2_scale.relations))

    def test_provider_and_plugin_entries_are_structured(self) -> None:
        provider = show_entry("provider:decision_evaluation_bridge", profile="framework-core")
        self.assertIsNotNone(provider)
        assert provider is not None
        self.assertEqual(provider.kind, "provider")
        self.assertEqual(str(provider.fields.get("provider_surface")), "bridge")
        self.assertIn("mlp_torch", tuple(provider.fields.get("applicable_presets", ())))
        self.assertIn("family:neural", tuple(provider.relations.get("families", ())))
        self.assertTrue(bool(provider.fields.get("supports_batch")))
        self.assertTrue(bool(provider.fields.get("supports_individual")))
        self.assertEqual(str(provider.fields.get("mount_point")), "problem_evaluation.provider")
        self.assertIn("decision_batch", tuple(provider.fields.get("contract_consumes", ())))
        self.assertIn("evaluation_result", tuple(provider.fields.get("contract_provides", ())))

        plugin = show_entry("plugin:experiment_tracker", profile="framework-core")
        self.assertIsNotNone(plugin)
        assert plugin is not None
        self.assertEqual(plugin.kind, "plugin")
        self.assertEqual(str(plugin.fields.get("plugin_surface")), "capability_registry")
        self.assertIn("sklearn_mlp", tuple(plugin.fields.get("applicable_presets", ())))
        self.assertIn("preset:mlp_torch", tuple(plugin.relations.get("presets", ())))
        self.assertIn("hook_events", dict(plugin.fields))
        self.assertIn("context_requires", dict(plugin.fields))
        self.assertEqual(str(plugin.fields.get("mount_point")), "capability_registry")
        self.assertEqual(tuple(plugin.fields.get("contract_requires", ())), tuple(plugin.fields.get("context_requires", ())))
        self.assertEqual(tuple(plugin.fields.get("contract_consumes", ())), tuple(plugin.fields.get("contract_requires", ())))

        filesystem_plugin = show_entry("plugin:trainer_state_checkpoint", profile="framework-core")
        self.assertIsNotNone(filesystem_plugin)
        assert filesystem_plugin is not None
        self.assertEqual(str(filesystem_plugin.fields.get("plugin_surface")), "flow_plugin")
        self.assertEqual(str(filesystem_plugin.fields.get("lifecycle_plane")), "flow")

    def test_catalog_db_target_info_supports_postgres_and_mysql_urls(self) -> None:
        pg = catalog_db_target_info("postgresql://demo:secret@localhost:5432/mlblack_catalog")
        self.assertEqual(str(pg.get("db_backend")), "postgresql")
        self.assertEqual(str(pg.get("db_driver")), "postgresql+psycopg")
        self.assertIn("***", str(pg.get("db_target")))
        self.assertNotIn("secret", str(pg.get("db_target")))

        mysql = catalog_db_target_info("mysql://demo:secret@localhost:3306/mlblack_catalog")
        self.assertEqual(str(mysql.get("db_backend")), "mysql")
        self.assertEqual(str(mysql.get("db_driver")), "mysql+pymysql")
        self.assertIn("***", str(mysql.get("db_target")))

    def test_catalog_scalar_schema_stays_mysql_portable(self) -> None:
        scalar_indexes = [
            idx
            for idx in _CATALOG_SCALARS.indexes
            if any(str(column.name) == "scalar_value" for column in idx.columns)
        ]
        self.assertEqual(scalar_indexes, [])

    def test_catalog_source_prefers_db_when_env_url_is_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_env.sqlite3"
            override_summary = self._materialize_overridden_catalog(db_path)
            env = {
                "MLBLACK_CATALOG_DB_URL": f"sqlite+pysqlite:///{db_path.as_posix()}",
                "MLBLACK_CATALOG_DB_MODE": "prefer",
            }
            with patch.dict(os.environ, env, clear=False):
                item = show_entry("preset:mlp_torch", profile="framework-core")
                self.assertIsNotNone(item)
                assert item is not None
                self.assertEqual(item.summary, override_summary)

                source = catalog_source_info(profile="framework-core")
                self.assertEqual(str(source.get("effective_source")), "db")
                self.assertEqual(str(source.get("source_mode")), "prefer")
            gc.collect()

    def test_catalog_source_mode_off_falls_back_to_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_off.sqlite3"
            override_summary = self._materialize_overridden_catalog(db_path)
            env = {
                "MLBLACK_CATALOG_DB_URL": f"sqlite+pysqlite:///{db_path.as_posix()}",
                "MLBLACK_CATALOG_DB_MODE": "off",
            }
            with patch.dict(os.environ, env, clear=False):
                item = show_entry("preset:mlp_torch", profile="framework-core")
                self.assertIsNotNone(item)
                assert item is not None
                self.assertNotEqual(item.summary, override_summary)

                source = catalog_source_info(profile="framework-core")
                self.assertEqual(str(source.get("effective_source")), "registry")
                self.assertEqual(str(source.get("source_mode")), "off")
            gc.collect()

    def test_catalog_source_can_resolve_db_toml_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_file.sqlite3"
            override_summary = self._materialize_overridden_catalog(db_path)
            config_path = Path(tmp) / "db.toml"
            config_path.write_text(
                "\n".join(
                    [
                        "[catalog_db]",
                        "enabled = true",
                        'mode = "only"',
                        'backend = "sqlite"',
                        f'path = "{db_path.as_posix()}"',
                        "readonly = true",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"MLBLACK_CATALOG_DB_CONFIG": str(config_path)}, clear=False):
                item = show_entry("preset:mlp_torch", profile="framework-core")
                self.assertIsNotNone(item)
                assert item is not None
                self.assertEqual(item.summary, override_summary)

                cfg = catalog_db_config_info()
                self.assertTrue(bool(cfg.get("enabled")))
                self.assertEqual(str(cfg.get("mode")), "only")
                self.assertTrue(bool(cfg.get("readonly")))
                self.assertEqual(str(cfg.get("db_backend")), "sqlite")
            gc.collect()

    def test_materialize_catalog_sqlite_alias_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "catalog_alias.sqlite3"
            payload = materialize_catalog_sqlite(str(db_path), profile="framework-core")
            self.assertEqual(str(payload.get("db_backend")), "sqlite")
            self.assertTrue(db_path.exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path

from catalog import catalog_source_info, catalog_ui_snapshot, list_entries, show_entry


def _write_project_catalog(root: Path) -> None:
    (root / ".mlblack-project").write_text("marker = mlblack-scaffold-project\n", encoding="utf-8")
    catalog_dir = root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    (catalog_dir / "entries.toml").write_text(
        "\n".join(
            [
                "[[entry]]",
                'key = "preset:demo_local"',
                'kind = "preset"',
                'name = "demo_local"',
                'summary = "project-local preset for catalog scope tests"',
                'tags = ["project", "demo"]',
                "",
                "[entry.fields]",
                'family = "neural"',
                'head = "point"',
                'title_zh = "本地预设"',
                'summary_zh = "项目本地预设条目"',
                "",
                "[entry.relations]",
                'family = ["family:neural"]',
            ]
        ),
        encoding="utf-8",
    )


def test_project_scope_catalog_listing_and_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "demo_project"
    root.mkdir()
    _write_project_catalog(root)

    info = catalog_source_info(profile="framework-core", scope="project", project_path=root)
    items = list_entries(profile="framework-core", scope="project", project_path=root, kind="preset")
    selected = show_entry("preset:demo_local", profile="framework-core", scope="project", project_path=root)
    snapshot = catalog_ui_snapshot(
        profile="framework-core",
        scope="project",
        project_path=root,
        kind="preset",
        selected_key="preset:demo_local",
    )

    assert info["scope"] == "project"
    assert info["project_found"] is True
    assert info["project_root"] == str(root.resolve())
    assert info["effective_source"] == "project"
    assert any(item.key == "preset:demo_local" for item in items)
    assert selected is not None
    assert selected.fields["title_zh"] == "本地预设"
    assert snapshot["selected"] is not None
    assert snapshot["selected"]["key"] == "preset:demo_local"


def test_project_scope_include_global_merges_framework_entries(tmp_path: Path) -> None:
    root = tmp_path / "demo_project"
    root.mkdir()
    _write_project_catalog(root)

    info = catalog_source_info(
        profile="framework-core",
        scope="project",
        project_path=root,
        include_global=True,
    )
    items = list_entries(
        profile="framework-core",
        scope="project",
        project_path=root,
        include_global=True,
        kind="preset",
    )

    keys = {item.key for item in items}
    assert info["effective_source"].startswith("project+")
    assert "preset:demo_local" in keys
    assert "preset:mlp_torch" in keys

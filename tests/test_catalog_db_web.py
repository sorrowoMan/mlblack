from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
import threading
from pathlib import Path
from urllib.request import urlopen

from mlblack.catalog import CatalogQuery, build_streamlit_command, materialize_catalog_db, query_catalog_db
from mlblack.catalog.facade import catalog_flow, catalog_neighbors, show_entry
from mlblack.catalog.registry import _DISCOVERY_PACKAGES, _entry_from_object, _is_catalog_component, get_catalog
from mlblack.catalog.relations import usage_profile
from mlblack.catalog.store import PostgresCatalogStore, resolve_catalog_store
from mlblack.catalog.web_app import catalog_web_payload, serve_catalog_web


def test_sqlite_catalog_db_materialize_and_query(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    result = materialize_catalog_db(db, refresh=True)

    assert result["backend"] == "sqlite"
    assert result["entries"] > 0

    query = query_catalog_db(CatalogQuery(kind="catalog", query="db", limit=10), db_path=str(db))
    keys = {entry.key for entry in query.entries}
    assert "catalog.query_db" in keys
    assert "catalog.materialize_db" in keys


def test_catalog_relations_and_flow_are_materialized(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    result = materialize_catalog_db(db, refresh=True)

    assert result["relations"] > 0
    store = resolve_catalog_store(str(db))
    relations = store.get_catalog_entry_relations("adapter.gradient_descent")
    assert relations["flow"]["current_stage"] == "optimization"
    assert relations["field_refs"]

    payload = show_entry("adapter.gradient_descent", db_path=str(db))
    assert payload["relations"]["usage"]["minimal_wiring"]
    assert catalog_flow("adapter.gradient_descent", db_path=str(db))["current_stage"] == "optimization"
    neighbors = catalog_neighbors("adapter.gradient_descent", db_path=str(db))
    assert "context_upstream" in neighbors


def test_neural_graph_codec_catalog_usage_is_component_specific(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    materialize_catalog_db(db, refresh=True)

    payload = show_entry("codec.neural_graph", db_path=str(db))
    usage = payload["relations"]["usage"]
    use_when = "\n".join(usage["use_when"])
    wiring = "\n".join(usage["minimal_wiring"])

    assert "NeuralGraphSpec" in use_when
    assert "flat parameter state" in use_when
    assert "neural.lowering" in use_when
    assert "NeuralGraphCodec.parameter_layout/init_values/decode" in wiring
    assert "backend.session" in usage["config_keys"]


def test_catalog_usage_is_useful_for_all_entries() -> None:
    catalog = get_catalog(refresh=True)
    bad_tokens = ("�", "闇€", "鐢", "浣跨", "瑙ｇ")

    for entry in catalog.list():
        usage = usage_profile(entry)
        assert usage["use_when"], entry.key
        assert usage["minimal_wiring"], entry.key
        assert usage["required_roles"], entry.key
        assert usage["config_keys"], entry.key

        joined = "\n".join(
            str(value)
            for section in ("use_when", "minimal_wiring", "required_roles", "config_keys", "notes")
            for value in usage.get(section, ())
        )
        assert "组件语义：" in joined or "架构位置：" in joined or entry.summary, entry.key
        assert not any(token in joined for token in bad_tokens), entry.key


def test_catalog_covers_component_shape_classes() -> None:
    catalog_imports = {entry.import_path for entry in get_catalog(refresh=True).list()}
    missing: list[tuple[str, str]] = []
    for package_name in _DISCOVERY_PACKAGES:
        try:
            package = importlib.import_module(package_name)
        except Exception:
            continue
        modules = [package]
        package_path = getattr(package, "__path__", None)
        if package_path:
            for modinfo in pkgutil.walk_packages(package_path, package.__name__ + "."):
                try:
                    modules.append(importlib.import_module(str(modinfo.name)))
                except Exception:
                    continue
        for module in modules:
            module_name = str(getattr(module, "__name__", ""))
            for name, obj in inspect.getmembers(module):
                if not _is_catalog_component(obj, module_name=module_name):
                    continue
                entry = _entry_from_object(obj, module_name=module_name, symbol_name=name)
                if entry.import_path not in catalog_imports:
                    missing.append((entry.key, entry.import_path))
    assert sorted(missing) == []


def test_postgres_catalog_store_resolver_surface() -> None:
    url = "postgresql://user:pass@127.0.0.1:5432/mlblack"
    try:
        store = resolve_catalog_store(url)
    except RuntimeError as exc:
        assert "psycopg" in str(exc)
        return
    assert isinstance(store, PostgresCatalogStore)
    assert store.backend == "postgresql"


def test_streamlit_dashboard_command_surface(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    command = build_streamlit_command(script_path=__file__, source="db", db_path=str(db), kind="catalog", query="streamlit", port=9876, headless=True)

    assert "-m" in command
    assert "streamlit" in command
    assert "--server.port" in command
    assert "9876" in command
    assert "--db-path" in command
    assert str(db) in command


def test_catalog_web_payload_and_http_api_read_db(tmp_path: Path) -> None:
    db = tmp_path / "catalog.sqlite"
    materialize_catalog_db(db, refresh=True)

    payload = catalog_web_payload({"source": "db", "db_path": str(db), "kind": "catalog", "q": "db", "limit": "5"})
    assert payload["entries"]

    server = serve_catalog_web(port=0, source="db", db_path=str(db))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        with urlopen(f"http://{host}:{port}/api/catalog?kind=catalog&q=db&limit=5", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        assert response.status == 200
        assert data["entries"]
        assert data["query"]["source"] == "db"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)

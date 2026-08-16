from .artifacts import export_artifact_html, load_artifact, render_artifact_markdown
from .backend_dashboard import backend_capability_matrix, export_backend_matrix_html, render_backend_matrix_markdown
from .dashboard import (
    catalog_summary,
    export_catalog_html,
    render_catalog_markdown,
)
from .dashboard_shell import (
    build_streamlit_command,
    launch_catalog_dashboard,
)

# alias for backward compatibility
render_streamlit_dashboard = launch_catalog_dashboard
from .experiment import (
    ExperimentQuery,
    ExperimentQueryResult,
    export_experiment_html,
    load_records,
    query_experiments,
    render_experiment_markdown,
    summarize_records,
)
from .facade import (
    catalog_flow,
    catalog_neighbors,
    catalog_schema,
    catalog_usage_matrix,
    list_entries,
    search_entries,
    show_entry,
)
from .query import CatalogQuery, CatalogQueryResult, build_catalog_deep_link, build_catalog_facets, query_catalog, query_catalog_db
from .relations import build_entry_relation_payload, build_relation_payload_index, flow_payload, relation_fields, usage_profile
from .registry import Catalog, CatalogEntry, CatalogRegistry, enrich_catalog_entry, get_catalog
from .store import (
    PostgresCatalogStore,
    SQLiteCatalogStore,
    DEFAULT_CATALOG_DB_PATH,
    catalog_db_summary,
    load_catalog_db,
    materialize_catalog_db,
    resolve_catalog_db_path,
    resolve_catalog_store,
)
from .web_app import catalog_summary_payload, catalog_web_payload, render_catalog_web_page, run_catalog_web, serve_catalog_web

__all__ = [
    "Catalog",
    "CatalogEntry",
    "CatalogRegistry",
    "CatalogQuery",
    "CatalogQueryResult",
    "ExperimentQuery",
    "ExperimentQueryResult",
    "DEFAULT_CATALOG_DB_PATH",
    "build_catalog_deep_link",
    "build_catalog_facets",
    "backend_capability_matrix",
    "build_streamlit_command",
    "build_catalog_webui_command",
    "catalog_db_summary",
    "catalog_summary",
    "catalog_summary_payload",
    "catalog_flow",
    "catalog_neighbors",
    "catalog_schema",
    "catalog_usage_matrix",
    "catalog_web_payload",
    "enrich_catalog_entry",
    "export_artifact_html",
    "export_backend_matrix_html",
    "export_catalog_html",
    "export_experiment_html",
    "get_catalog",
    "load_catalog_db",
    "load_records",
    "load_artifact",
    "list_entries",
    "materialize_catalog_db",
    "launch_catalog_dashboard",
    "launch_catalog_webui",
    "query_catalog",
    "query_catalog_db",
    "query_experiments",
    "relation_fields",
    "render_backend_matrix_markdown",
    "render_catalog_markdown",
    "render_streamlit_dashboard",
    "render_catalog_web_page",
    "render_artifact_markdown",
    "render_experiment_markdown",
    "resolve_catalog_db_path",
    "resolve_catalog_store",
    "PostgresCatalogStore",
    "SQLiteCatalogStore",
    "run_catalog_web",
    "search_entries",
    "serve_catalog_web",
    "show_entry",
    "summarize_records",
    "build_entry_relation_payload",
    "build_relation_payload_index",
    "flow_payload",
    "usage_profile",
]


def __getattr__(name: str):
    if name == "build_catalog_webui_command":
        from .webui import build_catalog_webui_command

        return build_catalog_webui_command
    if name == "launch_catalog_webui":
        from .webui import launch_catalog_webui

        return launch_catalog_webui
    raise AttributeError(name)

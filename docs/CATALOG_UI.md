# Catalog UI

`mlblack` catalog now exposes a structured field layer that can drive UI filters,
relation jumps, and detail panels.

## What the first UI-facing layer includes

Structured catalog kinds:

- `family`
- `preset`
- `head`
- `component`
- `provider`
- `plugin`

UI-ready helpers:

- `catalog_ui_snapshot(...)`
- `catalog_facets(...)`
- `catalog_neighbors(...)`
- `catalog_schema(...)`
- `field_values(...)`
- `materialize_catalog_db(...)`
- `catalog_db_summary(...)`
- `catalog_db_show_entry(...)`

These sit on top of the same registry already used by CLI discoverability.

## Standalone page

Open the catalog page with:

```powershell
python -m mlblack catalog ui --profile framework-core --kind preset
```

Use a DB-backed catalog page directly:

```powershell
python -m mlblack catalog ui --profile framework-core --kind preset --db-path postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/mlblack_catalog
python -m mlblack catalog ui --profile framework-core --kind preset --db-path mysql://root:YOUR_PASSWORD@127.0.0.1:33306/mlblack_catalog
```

Or let the page inherit routing from `catalog/db.toml` plus env mode:

```powershell
$env:MLBLACK_CATALOG_DB_URL = "postgresql://postgres:YOUR_PASSWORD@127.0.0.1:5432/mlblack_catalog"
$env:MLBLACK_CATALOG_DB_MODE = "only"
python -m mlblack catalog ui --profile framework-core --kind preset
```

Compatibility wrapper:

```powershell
streamlit run examples/run_catalog_dashboard.py -- --profile framework-core --kind preset
```

The page now supports:

- query search
- top `family / preset / head / component / provider / plugin` kind switch
- left-side field filters driven by structured facets
- center result list and result table with row selection
- right-side selected-entry detail panel
- right-side neighbor navigation via relation links
- raw UI snapshot inspection

## CLI helpers for UI/backend integration

```powershell
python -m mlblack catalog schema --profile framework-core --kind preset
python -m mlblack catalog values family --profile framework-core --kind preset
python -m mlblack catalog facets --profile framework-core --kind preset
python -m mlblack catalog snapshot --profile framework-core --kind preset --field family=neural
python -m mlblack catalog neighbors preset:mlp_torch --profile framework-core
```

## Materialize Catalog To SQL Store

If the catalog should behave more like a standalone indexed surface for UI or service use,
materialize it into a SQL catalog store first:

```powershell
python -m mlblack catalog db materialize --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db target --db-path postgresql://demo:secret@localhost:5432/mlblack_catalog
python -m mlblack catalog db summary --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db show preset:mlp_torch --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db list --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --field family=neural --format json
python -m mlblack catalog db search gradient_norm --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --format json
python -m mlblack catalog db facets --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --field family=neural --facet-field runtime_backend
python -m mlblack catalog db snapshot --db-path runs\catalog.sqlite3 --profile framework-core --kind preset --field family=neural --selected preset:mlp_torch
```

Current first-step SQL store scope:

- one row per structured catalog entry
- persisted `fields` / `relations` JSON payload
- db-backed `list/search/facets/snapshot` query surface
- persisted scalar index rows for filter/search acceleration
- per-profile summary/schema snapshot

This is the intended bridge toward a future UI that can read from a stable catalog database
instead of rebuilding the registry on every page launch.

Formal DB protocol, config precedence, and `source_mode` routing behavior are documented in
`docs/CATALOG_DB_PROTOCOL.md`.

Because the page delegates to the same facade/query surface as CLI:

- `--db-path` makes this page read the explicit SQL catalog target
- `MLBLACK_CATALOG_DB_URL` + `MLBLACK_CATALOG_DB_MODE` can switch the page into DB-backed mode
- once a profile is materialized, sqlite / PostgreSQL / MySQL all share the same list/search/facets/snapshot contract

## Intended integration pattern

For this page and future UI surfaces, the recommended layering is:

1. use `catalog_schema(...)` to discover fields by kind
2. use `catalog_facets(...)` to populate sidebar filters
3. use `catalog_ui_snapshot(...)` to fetch current result set + selected detail
4. use `catalog_neighbors(...)` when clicking relation-linked chips/cards

This keeps CLI, Streamlit, and future web/API consumers aligned on one field contract.

# Catalog DB Protocol

## Purpose

`mlblack` catalog now has two read surfaces:

- in-memory registry
- materialized SQL catalog store

This document defines the protocol that decides:

- how a catalog DB target is configured
- when reads come from registry vs DB
- how catalog materialization is performed
- which commands/functions are expected to stay symmetric across both surfaces

The goal is to make CLI, dashboard, tests, and future UI/service layers all read the
same catalog contract.

## Core Terms

### Registry

The default in-process catalog built from Python registry entries.

### Catalog DB

A materialized SQL store containing one or more catalog profiles.

Current backend protocol:

- sqlite
- PostgreSQL
- MySQL

Implementation uses SQLAlchemy as the common store layer.

### Profile

A named catalog projection such as:

- `default`
- `framework-core`

Profiles are materialized independently and may coexist in the same DB target.

### Source Mode

The read-routing mode used by facade APIs and CLI commands:

- `prefer`
- `only`
- `off`

## Supported DB Target Forms

Catalog DB targets may be supplied in either of these forms.

### 1. Plain sqlite file path

```powershell
runs\catalog.sqlite3
```

### 2. SQLAlchemy URL

```powershell
sqlite+pysqlite:///./runs/catalog.sqlite3
postgresql://user:pass@127.0.0.1:5432/mlblack_catalog
mysql://user:pass@127.0.0.1:3306/mlblack_catalog
```

Normalization rules:

- `postgresql://...` is normalized to driver `postgresql+psycopg`
- `mysql://...` is normalized to driver `mysql+pymysql`
- sqlite paths are normalized into `sqlite+pysqlite:///...`
- target info shown in CLI/API hides passwords

## Configuration Sources

Catalog DB config is resolved in this priority order.

### Explicit CLI/API target

If `--db-path` or `db_path=...` is provided for a read call, that explicit target wins.

### Environment variables

- `MLBLACK_CATALOG_DB_URL`
- `MLBLACK_CATALOG_DB_MODE`
- `MLBLACK_CATALOG_DB_CONFIG`
- `MLBLACK_CATALOG_DB_READONLY`

### TOML config file

The loader checks these files in order:

1. path from `MLBLACK_CATALOG_DB_CONFIG`
2. `./catalog/db.toml` under the current working directory
3. package template `mlblack/catalog/db.toml`

If no enabled config is found, facade reads stay on the in-memory registry unless an
explicit `--db-path` is passed.

## `catalog/db.toml` Schema

Primary form:

```toml
[catalog_db]
enabled = true
mode = "prefer"
readonly = false
url = "sqlite+pysqlite:///./runs/catalog.sqlite3"
```

Structured form:

```toml
[catalog_db]
enabled = true
mode = "prefer"
readonly = false
backend = "postgresql"
host = "127.0.0.1"
port = 5432
user = "postgres"
password = ""
database = "mlblack_catalog"
driver = "postgresql+psycopg"
```

Also supported:

- `backend = "sqlite"` with `path = "./runs/catalog.sqlite3"`
- `backend = "mysql"` with `driver = "mysql+pymysql"`
- nested form:

```toml
[catalog.db]
enabled = true
...
```

## Environment Variables

### `MLBLACK_CATALOG_DB_URL`

Provides the effective DB target directly.

If this variable is set, it overrides file target settings.

### `MLBLACK_CATALOG_DB_CONFIG`

Points to a custom `db.toml`.

### `MLBLACK_CATALOG_DB_MODE`

Overrides file `mode`.

Supported values:

- `prefer`
- `only`
- `off`
- `disabled` alias for `off`

### `MLBLACK_CATALOG_DB_READONLY`

Used only when the target comes from `MLBLACK_CATALOG_DB_URL`.

Truthy values:

- `1`
- `true`
- `yes`
- `on`

## Read Routing Semantics

Most user-facing catalog reads now go through the facade layer:

- `list_entries`
- `search_entries`
- `show_entry`
- `catalog_summary`
- `field_values`
- `catalog_schema`
- `catalog_neighbors`
- `catalog_facets`
- `catalog_ui_snapshot`
- `catalog_source_info`

### Mode `prefer`

Behavior:

- if DB config exists, target is reachable, and requested profile is materialized: read from DB
- otherwise: fall back to registry

This is the safest general mode for local development and UI exploration.

### Mode `only`

Behavior:

- DB must be configured or explicitly provided
- target must be reachable
- requested profile must already be materialized

Otherwise the read is an error.

Use this when you want CI, API, or UI to guarantee that it is reading the persisted catalog surface.

### Mode `off`

Behavior:

- always read from the in-memory registry
- config may still exist, but it is ignored for reads

Use this when you want to bypass DB routing entirely.

### Explicit `db_path`

If a read command is called with explicit `db_path` / `--db-path`:

- routing becomes DB-only for that call
- the requested profile must already be materialized
- `source_mode` effectively behaves like `only`

This avoids ambiguous mixed behavior.

## Materialization Semantics

Materialization writes one profile snapshot into the target store.

CLI:

```powershell
python -m mlblack catalog db materialize --db-path runs\catalog.sqlite3 --profile framework-core
```

Python:

```python
from catalog import materialize_catalog_db

payload = materialize_catalog_db("runs/catalog.sqlite3", profile="framework-core")
```

Write behavior:

- the target profile rows are replaced
- other profiles in the same DB are preserved
- summary and schema are persisted together with entries
- scalar index rows are persisted for field filter / facet / search support

Readonly rule:

- if `catalog db materialize` is called without explicit `--db-path`, the command may resolve
  target from config
- if that resolved config is marked `readonly = true`, implicit materialization is refused
- passing an explicit `--db-path` is the escape hatch when you intentionally want to write elsewhere

## Query Surface Symmetry

The SQL store intentionally mirrors the structured field API used by the registry layer.

DB-side primitives:

- `catalog_db_summary`
- `catalog_db_schema`
- `catalog_db_show_entry`
- `catalog_db_list_entries`
- `catalog_db_search_entries`
- `catalog_db_field_values`
- `catalog_db_neighbors`
- `catalog_db_facets`
- `catalog_db_ui_snapshot`

This symmetry is what lets the facade route UI/CLI reads without changing the higher-level contract.

## CLI Contract

Facade-backed commands:

```powershell
python -m mlblack catalog source --profile framework-core
python -m mlblack catalog list --profile framework-core --kind preset --source-mode prefer
python -m mlblack catalog search neural --profile framework-core --source-mode only
python -m mlblack catalog snapshot --profile framework-core --kind preset --field family=neural
```

DB-direct commands:

```powershell
python -m mlblack catalog db target
python -m mlblack catalog db summary --db-path runs\catalog.sqlite3 --profile framework-core
python -m mlblack catalog db values family --db-path runs\catalog.sqlite3 --profile framework-core --kind preset
python -m mlblack catalog db neighbors preset:mlp_torch --db-path runs\catalog.sqlite3 --profile framework-core
```

Practical split:

- `catalog ...` means "read through the routing protocol"
- `catalog db ...` means "operate on one explicit SQL catalog store"

## Dashboard Contract

Dashboard can be launched in two ways:

```powershell
python -m mlblack catalog ui --profile framework-core --kind preset
python -m mlblack catalog ui --profile framework-core --kind preset --db-path runs\catalog.sqlite3
```

Recommended usage:

- no `--db-path`: let facade routing decide based on config + mode
- with `--db-path`: force the page to read one known materialized target

The page should surface `catalog_source_info(...)` so users can see whether data came from:

- registry
- DB
- fallback from DB config to registry

## Backend Dependency Notes

Base dependency:

- `sqlalchemy>=2.0`

Optional extras:

- PostgreSQL: `psycopg[binary]>=3.1`
- MySQL: `pymysql>=1.1`

Meaning:

- code path supports sqlite / PostgreSQL / MySQL
- live integration still depends on the relevant driver package and an actual running server

## Recommended Workflows

### Local single-user development

Use sqlite and `prefer`:

```toml
[catalog_db]
enabled = true
mode = "prefer"
backend = "sqlite"
path = "./runs/catalog.sqlite3"
```

### Shared service / UI backend

Use PostgreSQL and materialize `framework-core` into a team-visible DB.

Prefer `only` if the service must never silently fall back to registry.

### CI / regression checks

Use `only` so missing materialization becomes a hard failure.

This is the cleanest way to verify that exported catalog state is really what the caller is reading.

## Current Non-Goals

This protocol does not yet promise:

- migration tooling between schema versions
- concurrent write orchestration between multiple materializers
- database-specific tuning beyond the common SQLAlchemy layer

Those can be added later without changing the high-level read/write protocol defined here.

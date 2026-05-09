from __future__ import annotations

# Compatibility shim: the catalog DB layer now lives in sql_store.py and supports
# sqlite paths plus SQLAlchemy URLs for PostgreSQL/MySQL.
from .sql_store import *  # noqa: F401,F403


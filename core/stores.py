"""
Forwarding module for stores.

This module re-exports from blackbase for seamless migration.
"""

from blackbase.context import (
    ContextStore,
    InMemoryContextStore,
    SnapshotStore,
    InMemorySnapshotStore,
    SnapshotRecord,
    create_context_store,
    create_snapshot_store,
)

__all__ = [
    "ContextStore",
    "InMemoryContextStore",
    "SnapshotStore",
    "InMemorySnapshotStore",
    "SnapshotRecord",
    "create_context_store",
    "create_snapshot_store",
]

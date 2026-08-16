"""Forwarding module for capability base class.

This module re-exports from blackbase for seamless migration.
The Capability class is now provided by blackbase.adapters.mlblack.plugin
as a Plugin subclass with mlblack-style lifecycle hooks.

For new code, prefer using blackbase.plugin.Plugin directly.
"""

from __future__ import annotations

from blackbase.adapters.mlblack.plugin import (
    Capability,
    CapabilityPluginAdapter,
)

__all__ = [
    "Capability",
    "CapabilityPluginAdapter",
]

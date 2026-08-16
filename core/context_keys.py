"""
Forwarding module for context keys.

This module re-exports from blackbase for seamless migration.
"""

from blackbase.context import (
    CONTEXT_KEY_ALIASES,
    CONTEXT_KEY_SET,
    METRIC_FALLBACKS,
    METRIC_KEYS,
    normalize_context_key,
    normalize_context_keys,
    register_context_keys,
    unknown_context_keys,
    validate_context_keys,
)

REGISTERED_CONTEXT_KEYS = CONTEXT_KEY_SET


__all__ = [
    "CONTEXT_KEY_ALIASES",
    "CONTEXT_KEY_SET",
    "METRIC_FALLBACKS",
    "METRIC_KEYS",
    "REGISTERED_CONTEXT_KEYS",
    "normalize_context_key",
    "normalize_context_keys",
    "register_context_keys",
    "unknown_context_keys",
    "validate_context_keys",
]

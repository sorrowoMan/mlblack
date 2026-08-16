"""
Forwarding module for resources.

This module re-exports from blackbase for seamless migration.
"""

from blackbase.resources import (
    InMemoryLeaseStore,
    ResourceAllocator,
    ResourceContext,
    ResourceEvent,
    ResourceAudit,
    ResourceLease,
    ResourceOffer,
    ResourcePolicy,
    ResourceRequest,
    coerce_resource_context,
)

__all__ = [
    "InMemoryLeaseStore",
    "ResourceAllocator",
    "ResourceContext",
    "ResourceEvent",
    "ResourceAudit",
    "ResourceLease",
    "ResourceOffer",
    "ResourcePolicy",
    "ResourceRequest",
    "coerce_resource_context",
]

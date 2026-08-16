"""
Forwarding module for compute pool.

This module re-exports from blackbase for seamless migration.
"""

from blackbase.resources import (
    WorkerDescriptor,
    ResourceOffer,
    build_local_worker_descriptor,
    detect_local_resource_offer,
    PoolScheduler,
    PoolTask,
    PoolResult,
    PoolTaskResult,
)

__all__ = [
    "WorkerDescriptor",
    "ResourceOffer",
    "build_local_worker_descriptor",
    "detect_local_resource_offer",
    "PoolScheduler",
    "PoolTask",
    "PoolResult",
    "PoolTaskResult",
]

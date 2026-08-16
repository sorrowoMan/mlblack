"""
Forwarding module for resources package.

This module re-exports from blackbase for seamless migration.
"""

from blackbase.resources import (
    DataRef,
    InMemoryLeaseStore,
    ResourceOffer,
    ResourceAllocator,
    ResourceLease,
    ResourcePolicy,
    ResourceRequest,
    ResourceRequirement,
    WorkerDescriptor,
    TaskEnvelope,
    TaskResult,
    ScheduledTask,
    ResourceContext,
    ResourceEvent,
    ResourceAudit,
    coerce_resource_context,
    detect_total_memory_mb,
    detect_cuda_devices,
    detect_local_resource_offer,
    build_local_worker_descriptor,
    PoolScheduler,
    PoolTask,
    PoolResult,
    PoolTaskResult,
)


__all__ = [
    "DataRef",
    "InMemoryLeaseStore",
    "ResourceOffer",
    "ResourceAllocator",
    "ResourceLease",
    "ResourcePolicy",
    "ResourceRequest",
    "ResourceRequirement",
    "WorkerDescriptor",
    "TaskEnvelope",
    "TaskResult",
    "ScheduledTask",
    "ResourceContext",
    "ResourceEvent",
    "ResourceAudit",
    "coerce_resource_context",
    "detect_total_memory_mb",
    "detect_cuda_devices",
    "detect_local_resource_offer",
    "build_local_worker_descriptor",
    "PoolScheduler",
    "PoolTask",
    "PoolResult",
    "PoolTaskResult",
]

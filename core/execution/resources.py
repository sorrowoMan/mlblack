from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .registry import (
    ExecutionBackendRegistry,
    ExecutionDeviceRegistry,
    normalize_execution_device_token,
)


@dataclass(frozen=True)
class ExecutionResourceOffer:
    threads: int = 1
    cuda_devices: tuple[str, ...] = ()
    mps_devices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "threads", max(1, int(self.threads)))
        object.__setattr__(self, "cuda_devices", _normalize_offer_tokens(self.cuda_devices, kind="cuda"))
        object.__setattr__(self, "mps_devices", _normalize_offer_tokens(self.mps_devices, kind="mps"))

    @property
    def gpus(self) -> int:
        return int(len(self.cuda_devices) + len(self.mps_devices))

    @property
    def device_tokens(self) -> tuple[str, ...]:
        return tuple(self.cuda_devices) + tuple(self.mps_devices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threads": int(self.threads),
            "cuda_devices": [str(x) for x in self.cuda_devices],
            "mps_devices": [str(x) for x in self.mps_devices],
            "device_tokens": [str(x) for x in self.device_tokens],
            "gpus": int(self.gpus),
        }


@dataclass(frozen=True)
class ExecutionResourceRequest:
    threads: int = 1
    backend: str = "serial"
    label: str = ""
    device_tokens: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        threads = int(self.threads)
        if threads < 0:
            raise ValueError("ExecutionResourceRequest.threads must be >= 0")
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "backend", str(self.backend or "serial").strip().lower() or "serial")
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "device_tokens", _normalize_request_tokens(self.device_tokens))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def cuda_devices(self) -> tuple[str, ...]:
        return tuple(token for token in self.device_tokens if _token_kind(token) == "cuda")

    @property
    def mps_devices(self) -> tuple[str, ...]:
        return tuple(token for token in self.device_tokens if _token_kind(token) == "mps")

    @property
    def gpus(self) -> int:
        return int(len(self.cuda_devices) + len(self.mps_devices))

    def as_dict(self) -> dict[str, Any]:
        return {
            "threads": int(self.threads),
            "backend": str(self.backend),
            "label": str(self.label),
            "device_tokens": [str(x) for x in self.device_tokens],
            "cuda_devices": [str(x) for x in self.cuda_devices],
            "mps_devices": [str(x) for x in self.mps_devices],
            "gpus": int(self.gpus),
            "metadata": dict(self.metadata),
        }


EXECUTION_RESOURCE_GRANT_KEY = "execution_resource_grant"
EXECUTION_USAGE_REPORTS_KEY = "execution_usage_reports"


@dataclass(frozen=True)
class ExecutionResourceGrant:
    phase: str
    threads: int
    backend: str = "serial"
    label: str = ""
    device_tokens: tuple[str, ...] = ()
    request_label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        threads = int(self.threads)
        if threads < 1:
            raise ValueError("ExecutionResourceGrant.threads must be >= 1")
        object.__setattr__(self, "phase", str(self.phase or "phase").strip() or "phase")
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "backend", str(self.backend or "serial").strip().lower() or "serial")
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "request_label", str(self.request_label or "").strip())
        object.__setattr__(self, "device_tokens", _normalize_request_tokens(self.device_tokens))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def cuda_devices(self) -> tuple[str, ...]:
        return tuple(token for token in self.device_tokens if _token_kind(token) == "cuda")

    @property
    def mps_devices(self) -> tuple[str, ...]:
        return tuple(token for token in self.device_tokens if _token_kind(token) == "mps")

    @property
    def gpus(self) -> int:
        return int(len(self.cuda_devices) + len(self.mps_devices))

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": str(self.phase),
            "threads": int(self.threads),
            "backend": str(self.backend),
            "label": str(self.label),
            "request_label": str(self.request_label),
            "device_tokens": [str(x) for x in self.device_tokens],
            "cuda_devices": [str(x) for x in self.cuda_devices],
            "mps_devices": [str(x) for x in self.mps_devices],
            "gpus": int(self.gpus),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionUsageReport:
    phase: str
    label: str
    backend: str
    granted_threads: int
    peak_threads: int
    used_threads: int | None = None
    request_label: str = ""
    device_tokens: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        granted_threads = int(self.granted_threads)
        peak_threads = int(self.peak_threads)
        used_threads = None if self.used_threads is None else int(self.used_threads)
        if granted_threads < 1:
            raise ValueError("ExecutionUsageReport.granted_threads must be >= 1")
        if peak_threads < 0:
            raise ValueError("ExecutionUsageReport.peak_threads must be >= 0")
        if used_threads is not None and used_threads < 0:
            raise ValueError("ExecutionUsageReport.used_threads must be >= 0")
        object.__setattr__(self, "phase", str(self.phase or "phase").strip() or "phase")
        object.__setattr__(self, "label", str(self.label or "").strip())
        object.__setattr__(self, "backend", str(self.backend or "serial").strip().lower() or "serial")
        object.__setattr__(self, "granted_threads", granted_threads)
        object.__setattr__(self, "peak_threads", peak_threads)
        object.__setattr__(self, "used_threads", used_threads)
        object.__setattr__(self, "request_label", str(self.request_label or "").strip())
        object.__setattr__(self, "device_tokens", _normalize_request_tokens(self.device_tokens))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": str(self.phase),
            "label": str(self.label),
            "backend": str(self.backend),
            "granted_threads": int(self.granted_threads),
            "peak_threads": int(self.peak_threads),
            "used_threads": None if self.used_threads is None else int(self.used_threads),
            "request_label": str(self.request_label),
            "device_tokens": [str(x) for x in self.device_tokens],
            "metadata": dict(self.metadata),
        }


class ExecutionBudgetError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        phase: str,
        offer: ExecutionResourceOffer,
        total_request: ExecutionResourceRequest | None = None,
        requests: Sequence[ExecutionResourceRequest] = (),
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = str(phase)
        self.offer = offer
        self.total_request = total_request
        self.requests = tuple(requests)
        self.detail = {} if detail is None else dict(detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": str(self.phase),
            "message": str(self),
            "offer": self.offer.as_dict(),
            "total_request": None if self.total_request is None else self.total_request.as_dict(),
            "requests": [request.as_dict() for request in self.requests],
            "detail": dict(self.detail),
        }


def detect_local_execution_offer(
    *,
    device_registry: ExecutionDeviceRegistry | None = None,
    torch_module: Any | None = None,
) -> ExecutionResourceOffer:
    registry = device_registry or ExecutionDeviceRegistry.global_registry()
    return ExecutionResourceOffer(
        threads=_detect_available_threads(),
        cuda_devices=tuple(registry.discover("cuda", torch_module=torch_module)),
        mps_devices=tuple(registry.discover("mps", torch_module=torch_module)),
    )


def coerce_execution_resource_request(
    value: ExecutionResourceRequest | Mapping[str, Any] | None = None,
    *,
    threads: int | None = None,
    backend: str = "serial",
    label: str = "",
    device_tokens: Sequence[str | int] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionResourceRequest:
    if isinstance(value, ExecutionResourceRequest):
        if not label or value.label:
            return value
        return ExecutionResourceRequest(
            threads=int(value.threads),
            backend=str(value.backend),
            label=str(label),
            device_tokens=tuple(value.device_tokens),
            metadata=dict(value.metadata),
        )

    if value is None:
        payload: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError("execution resource request must be ExecutionResourceRequest, mapping, or None")

    request_threads = payload.get("threads", threads if threads is not None else 1)
    request_backend = payload.get("backend", backend)
    request_label = payload.get("label", label)
    request_devices = payload.get("device_tokens", payload.get("devices", payload.get("gpu_devices", device_tokens)))
    request_metadata = payload.get("metadata", metadata or {})
    return ExecutionResourceRequest(
        threads=int(request_threads),
        backend=str(request_backend),
        label=str(request_label),
        device_tokens=tuple(request_devices),
        metadata=dict(request_metadata),
    )


def coerce_execution_resource_grant(
    value: ExecutionResourceGrant | ExecutionResourceRequest | Mapping[str, Any],
    *,
    phase: str = "",
    label: str = "",
) -> ExecutionResourceGrant:
    if isinstance(value, ExecutionResourceGrant):
        return value
    if isinstance(value, ExecutionResourceRequest):
        return issue_execution_resource_grant(
            value,
            phase=(phase or "phase"),
            label=(label or str(value.label or "grant")),
        )
    if not isinstance(value, Mapping):
        raise TypeError("execution resource grant must be ExecutionResourceGrant, ExecutionResourceRequest, or mapping")

    payload = dict(value)
    if "grant" in payload and isinstance(payload.get("grant"), Mapping):
        payload = dict(payload.get("grant") or {})
    if "request" in payload and isinstance(payload.get("request"), Mapping):
        request = coerce_execution_resource_request(payload.get("request"))
        payload.setdefault("threads", int(request.threads))
        payload.setdefault("backend", str(request.backend))
        payload.setdefault("request_label", str(request.label))
        payload.setdefault("device_tokens", tuple(request.device_tokens))

    return ExecutionResourceGrant(
        phase=str(payload.get("phase", phase or "phase")),
        threads=int(payload.get("threads", 1)),
        backend=str(payload.get("backend", "serial")),
        label=str(payload.get("label", label or "")),
        request_label=str(payload.get("request_label", "")),
        device_tokens=tuple(payload.get("device_tokens", payload.get("devices", ()))),
        metadata=dict(payload.get("metadata", {})),
    )


def issue_execution_resource_grant(
    request: ExecutionResourceRequest | Mapping[str, Any],
    *,
    phase: str,
    label: str = "",
    max_threads: int | None = None,
    backend: str | None = None,
    device_tokens: Sequence[str | int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionResourceGrant:
    req = coerce_execution_resource_request(request)
    granted_threads = int(req.threads if max_threads is None else min(int(req.threads), int(max_threads)))
    if granted_threads < 1:
        raise ValueError("execution resource grant must approve at least one thread")
    granted_tokens = tuple(req.device_tokens) if device_tokens is None else tuple(device_tokens)
    grant_metadata = dict(req.metadata)
    if metadata:
        grant_metadata.update(dict(metadata))
    grant_metadata.setdefault("request", req.as_dict())
    return ExecutionResourceGrant(
        phase=str(phase or "phase"),
        threads=int(granted_threads),
        backend=str(req.backend if backend is None else backend),
        label=str(label or req.label or "grant"),
        request_label=str(req.label),
        device_tokens=granted_tokens,
        metadata=grant_metadata,
    )


def constrain_execution_offer_to_grant(
    offer: ExecutionResourceOffer,
    grant: ExecutionResourceGrant | Mapping[str, Any] | None,
) -> ExecutionResourceOffer:
    if grant is None:
        return offer
    normalized_grant = coerce_execution_resource_grant(grant)
    return ExecutionResourceOffer(
        threads=max(1, min(int(offer.threads), int(normalized_grant.threads))),
        cuda_devices=_constrain_offer_tokens(tuple(offer.cuda_devices), tuple(normalized_grant.cuda_devices)),
        mps_devices=_constrain_offer_tokens(tuple(offer.mps_devices), tuple(normalized_grant.mps_devices)),
    )


def build_execution_usage_report(
    grant: ExecutionResourceGrant | Mapping[str, Any],
    *,
    label: str = "",
    peak_threads: int,
    used_threads: int | None = None,
    backend: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ExecutionUsageReport:
    normalized_grant = coerce_execution_resource_grant(grant)
    report_metadata = dict(normalized_grant.metadata)
    if metadata:
        report_metadata.update(dict(metadata))
    return ExecutionUsageReport(
        phase=str(normalized_grant.phase),
        label=str(label or normalized_grant.label or "usage"),
        backend=str(normalized_grant.backend if backend is None else backend),
        granted_threads=int(normalized_grant.threads),
        peak_threads=int(peak_threads),
        used_threads=(None if used_threads is None else int(used_threads)),
        request_label=str(normalized_grant.request_label),
        device_tokens=tuple(normalized_grant.device_tokens),
        metadata=report_metadata,
    )


def resolve_phase_worker_count(
    requested: int | None,
    *,
    n_tasks: int,
    offer: ExecutionResourceOffer | None = None,
) -> int:
    task_count = max(0, int(n_tasks))
    if task_count <= 0:
        return 0
    if requested is None:
        if offer is None:
            return max(1, task_count)
        return max(1, min(task_count, int(offer.threads)))
    workers = int(requested)
    if workers <= 0:
        raise ValueError("phase worker count must be a positive integer")
    return max(1, min(workers, task_count))


def sum_execution_resource_requests(
    requests: Sequence[ExecutionResourceRequest | Mapping[str, Any]],
    *,
    label: str = "",
) -> ExecutionResourceRequest:
    normalized = tuple(
        coerce_execution_resource_request(item, label=(label if len(tuple(requests)) == 1 else ""))
        for item in tuple(requests)
    )
    backends = {str(item.backend) for item in normalized if str(item.backend).strip()}
    merged_devices: list[str] = []
    merged_metadata: dict[str, Any] = {"requests": [item.as_dict() for item in normalized]}
    for item in normalized:
        merged_devices.extend(str(token) for token in item.device_tokens)
    return ExecutionResourceRequest(
        threads=sum(int(item.threads) for item in normalized),
        backend=(next(iter(backends)) if len(backends) == 1 else "mixed"),
        label=str(label),
        device_tokens=tuple(merged_devices),
        metadata=merged_metadata,
    )


def nested_total_execution_request(
    outer: ExecutionResourceRequest | Mapping[str, Any],
    inner: ExecutionResourceRequest | Mapping[str, Any] | None,
    *,
    fanout: int | None = None,
    label: str = "",
) -> ExecutionResourceRequest:
    outer_req = coerce_execution_resource_request(outer)
    inner_req = None if inner is None else coerce_execution_resource_request(inner)
    if inner_req is None:
        return outer_req if not label else ExecutionResourceRequest(
            threads=int(outer_req.threads),
            backend=str(outer_req.backend),
            label=str(label),
            device_tokens=tuple(outer_req.device_tokens),
            metadata=dict(outer_req.metadata),
        )
    replicas = int(fanout if fanout is not None else max(1, int(outer_req.threads)))
    inner_requests = [
        ExecutionResourceRequest(
            threads=int(inner_req.threads),
            backend=str(inner_req.backend),
            label=(str(inner_req.label) or "inner"),
            device_tokens=tuple(inner_req.device_tokens),
            metadata=dict(inner_req.metadata),
        )
        for _ in range(max(0, replicas))
    ]
    return sum_execution_resource_requests((outer_req, *inner_requests), label=label or str(outer_req.label))


def assert_phase_resource_budget(
    phase: str,
    requests: Sequence[ExecutionResourceRequest | Mapping[str, Any]],
    *,
    offer: ExecutionResourceOffer,
    backend_registry: ExecutionBackendRegistry | None = None,
) -> dict[str, Any]:
    phase_name = str(phase or "phase")
    registry = backend_registry or ExecutionBackendRegistry.global_registry()
    normalized = tuple(coerce_execution_resource_request(item) for item in tuple(requests))
    errors: list[str] = []

    for request in normalized:
        try:
            backend_spec = registry.resolve(request.backend)
        except Exception as exc:
            errors.append(
                f"{phase_name} request '{request.label or request.backend}' uses unsupported backend "
                f"'{request.backend}': {exc}"
            )
            continue
        unsupported_kinds = [
            kind for kind in _request_device_kinds(request) if kind not in set(backend_spec.supported_device_kinds)
        ]
        if unsupported_kinds:
            errors.append(
                f"{phase_name} request '{request.label or request.backend}' asks backend '{request.backend}' "
                f"for unsupported device kinds {sorted(set(unsupported_kinds))}"
            )

    total_request = sum_execution_resource_requests(normalized, label=phase_name)
    if int(total_request.threads) > int(offer.threads):
        errors.append(f"{phase_name} threads over budget: need={int(total_request.threads)}, offer={int(offer.threads)}")
    errors.extend(_device_budget_errors(phase_name, kind="cuda", requested=total_request.cuda_devices, offered=offer.cuda_devices))
    errors.extend(_device_budget_errors(phase_name, kind="mps", requested=total_request.mps_devices, offered=offer.mps_devices))

    if errors:
        raise ExecutionBudgetError(
            "; ".join(errors),
            phase=phase_name,
            offer=offer,
            total_request=total_request,
            requests=normalized,
            detail={"errors": list(errors)},
        )

    return {
        "phase": phase_name,
        "offer": offer.as_dict(),
        "requests": [request.as_dict() for request in normalized],
        "total_request": total_request.as_dict(),
    }


def clamp_worker_count(
    requested: int | None,
    *,
    n_tasks: int,
    offer: ExecutionResourceOffer,
) -> int:
    if n_tasks <= 1:
        return 1
    capacity = max(1, int(offer.threads))
    if requested is None:
        return max(1, min(int(n_tasks), capacity))
    workers = int(requested)
    if workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    return max(1, min(workers, int(n_tasks), capacity))


def filter_available_device_tokens(
    devices: Sequence[str | int],
    *,
    kind: str = "cuda",
    device_registry: ExecutionDeviceRegistry | None = None,
    torch_module: Any | None = None,
) -> tuple[str, ...]:
    registry = device_registry or ExecutionDeviceRegistry.global_registry()
    normalized_requested: list[str] = []
    for raw in tuple(devices):
        token = normalize_execution_device_token(raw)
        if token not in normalized_requested:
            normalized_requested.append(token)

    available = tuple(registry.discover(kind, torch_module=torch_module))
    if not available:
        return tuple()
    allowed = set(available)
    return tuple(token for token in normalized_requested if token in allowed)


def _normalize_offer_tokens(tokens: Sequence[str | int], *, kind: str) -> tuple[str, ...]:
    expected = str(kind).strip().lower()
    out: list[str] = []
    for raw in tuple(tokens):
        token = normalize_execution_device_token(raw)
        if _token_kind(token) != expected:
            raise ValueError(f"ExecutionResourceOffer received non-{expected} token: {raw}")
        if token not in out:
            out.append(token)
    return tuple(out)


def _normalize_request_tokens(tokens: Sequence[str | int]) -> tuple[str, ...]:
    out: list[str] = []
    for raw in tuple(tokens):
        token = normalize_execution_device_token(raw)
        kind = _token_kind(token)
        if kind not in {"cuda", "mps"}:
            continue
        out.append(token)
    return tuple(out)


def _constrain_offer_tokens(offered: Sequence[str], granted: Sequence[str]) -> tuple[str, ...]:
    if not granted:
        return tuple(str(token) for token in tuple(offered))
    offered_tokens = [str(token) for token in tuple(offered)]
    resolved: list[str] = []
    for raw in tuple(granted):
        token = str(raw)
        if token == "cuda":
            next_cuda = next((item for item in offered_tokens if item.startswith("cuda:") and item not in resolved), None)
            if next_cuda is not None:
                resolved.append(next_cuda)
            continue
        if token == "mps":
            if "mps" in offered_tokens and "mps" not in resolved:
                resolved.append("mps")
            continue
        if token in offered_tokens and token not in resolved:
            resolved.append(token)
    return tuple(resolved)


def _token_kind(token: str) -> str:
    key = str(token).strip().lower()
    if key == "mps":
        return "mps"
    if key == "cuda" or key.startswith("cuda:"):
        return "cuda"
    return "cpu"


def _request_device_kinds(request: ExecutionResourceRequest) -> tuple[str, ...]:
    kinds: list[str] = []
    for token in request.device_tokens:
        kind = _token_kind(token)
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def _device_budget_errors(
    phase: str,
    *,
    kind: str,
    requested: Sequence[str],
    offered: Sequence[str],
) -> list[str]:
    if not requested:
        return []

    errors: list[str] = []
    offered_tokens = tuple(str(token) for token in tuple(offered))
    concrete_capacity = Counter(token for token in offered_tokens if ":" in token or token == "mps")
    requested_tokens = tuple(str(token) for token in tuple(requested))
    requested_total = len(requested_tokens)
    concrete_requested = Counter(token for token in requested_tokens if token != kind)
    abstract_requested = sum(1 for token in requested_tokens if token == kind)

    if requested_total > len(offered_tokens):
        errors.append(f"{phase} {kind} devices over budget: need={requested_total}, offer={len(offered_tokens)}")

    for token, count in concrete_requested.items():
        if token not in concrete_capacity:
            errors.append(f"{phase} requested unavailable {kind} device '{token}'")
            continue
        if int(count) > int(concrete_capacity[token]):
            errors.append(
                f"{phase} requested {kind} device '{token}' {int(count)} times but only "
                f"{int(concrete_capacity[token])} slot is available"
            )

    concrete_claims = sum(int(count) for count in concrete_requested.values())
    remaining_capacity = max(0, len(offered_tokens) - concrete_claims)
    if abstract_requested > remaining_capacity:
        errors.append(
            f"{phase} abstract {kind} claims over budget: need_additional={int(abstract_requested)}, "
            f"remaining_offer={int(remaining_capacity)}"
        )

    return errors


def _detect_available_threads() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, int(len(os.sched_getaffinity(0))))
    except Exception:
        pass
    return max(1, int(os.cpu_count() or 1))


__all__ = [
    "EXECUTION_RESOURCE_GRANT_KEY",
    "EXECUTION_USAGE_REPORTS_KEY",
    "ExecutionBudgetError",
    "ExecutionResourceGrant",
    "ExecutionResourceOffer",
    "ExecutionResourceRequest",
    "ExecutionUsageReport",
    "assert_phase_resource_budget",
    "build_execution_usage_report",
    "clamp_worker_count",
    "coerce_execution_resource_grant",
    "coerce_execution_resource_request",
    "constrain_execution_offer_to_grant",
    "detect_local_execution_offer",
    "filter_available_device_tokens",
    "issue_execution_resource_grant",
    "nested_total_execution_request",
    "resolve_phase_worker_count",
    "sum_execution_resource_requests",
]

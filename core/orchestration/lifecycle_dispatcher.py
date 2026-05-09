from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence

from .lifecycle_events import describe_lifecycle_event_table, resolve_lifecycle_event


@dataclass
class LifecycleDispatcher:
    """Unified lifecycle dispatcher for flow capabilities and runtime hooks."""

    strict: bool = False
    _participants: list[tuple[int, Any]] = field(default_factory=list)
    _next_ordinal: int = 0
    _profile: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register(self, participant: Any) -> None:
        ordinal = int(self._next_ordinal)
        self._participants.append((ordinal, participant))
        self._next_ordinal += 1

        key = self._profile_key(participant, ordinal)
        if key not in self._profile:
            self._profile[key] = {
                "name": self._participant_name(participant),
                "priority": self._participant_priority(participant),
                "enabled": self._participant_enabled(participant),
                "is_algorithmic": self._participant_is_algorithmic(participant),
                "total_s": 0.0,
                "events": {},
            }

    def list_participants(self) -> tuple[Any, ...]:
        return tuple(participant for _, participant in sorted(self._participants, key=self._sort_key))

    def emit(self, event: str, *args: Any) -> None:
        key = str(event).strip()
        if not key:
            return

        spec = resolve_lifecycle_event(key)
        if spec is None:
            bindings = ((key, tuple(args)),)
        else:
            bindings = []
            for binding in spec.hook_bindings:
                bindings.append((str(binding.hook_name), tuple(binding.adapter(tuple(args)))))

        for ordinal, participant in sorted(self._participants, key=self._sort_key):
            if not self._participant_enabled(participant):
                continue

            t0 = time.perf_counter()
            try:
                for hook_name, hook_args in bindings:
                    hook = getattr(participant, hook_name, None)
                    if not callable(hook):
                        continue
                    hook(*hook_args)
            except Exception as exc:
                if self.strict:
                    raise
                warnings.warn(
                    f"Lifecycle participant '{self._participant_name(participant)}' failed in {key}: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            finally:
                dt = max(0.0, float(time.perf_counter() - t0))
                profile = self._profile.get(self._profile_key(participant, ordinal))
                if profile is None:
                    continue
                profile["enabled"] = self._participant_enabled(participant)
                profile["priority"] = self._participant_priority(participant)
                profile["is_algorithmic"] = self._participant_is_algorithmic(participant)
                profile["total_s"] = float(profile.get("total_s", 0.0)) + dt
                events = profile.setdefault("events", {})
                row = events.setdefault(key, {"calls": 0, "total_s": 0.0})
                row["calls"] = int(row.get("calls", 0)) + 1
                row["total_s"] = float(row.get("total_s", 0.0)) + dt

    dispatch = emit

    def build_report(self, *, participants: Sequence[Any] | None = None) -> Dict[str, Any]:
        selected_ids = None if participants is None else {id(participant) for participant in tuple(participants)}
        rows: list[Mapping[str, Any]] = []
        for ordinal, participant in sorted(self._participants, key=self._sort_key):
            if selected_ids is not None and id(participant) not in selected_ids:
                continue
            profile = self._profile.get(self._profile_key(participant, ordinal), {})
            contract_fn = getattr(participant, "get_context_contract", None)
            context_contract = {}
            if callable(contract_fn):
                try:
                    value = contract_fn()
                    if isinstance(value, Mapping):
                        context_contract = dict(value)
                except Exception:
                    context_contract = {}
            rows.append(
                {
                    "name": self._participant_name(participant),
                    "priority": self._participant_priority(participant),
                    "enabled": self._participant_enabled(participant),
                    "is_algorithmic": self._participant_is_algorithmic(participant),
                    "context_contract": context_contract,
                    "profile": {
                        "total_s": float(profile.get("total_s", 0.0)),
                        "events": dict(profile.get("events", {})),
                    },
                }
            )
        return {
            "strict": bool(self.strict),
            "count": int(len(rows)),
            "items": rows,
        }

    def describe_event_table(self) -> tuple[dict[str, Any], ...]:
        return describe_lifecycle_event_table()

    def _sort_key(self, item: tuple[int, Any]) -> tuple[int, str, int]:
        ordinal, participant = item
        return (
            self._participant_priority(participant),
            self._participant_name(participant),
            int(ordinal),
        )

    @staticmethod
    def _participant_name(participant: Any) -> str:
        value = getattr(participant, "name", type(participant).__name__)
        return str(value)

    @staticmethod
    def _participant_priority(participant: Any) -> int:
        value = getattr(participant, "priority", 0)
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _participant_enabled(participant: Any) -> bool:
        value = getattr(participant, "enabled", True)
        return bool(value)

    @staticmethod
    def _participant_is_algorithmic(participant: Any) -> bool:
        value = getattr(participant, "is_algorithmic", False)
        return bool(value)

    @staticmethod
    def _profile_key(participant: Any, ordinal: int) -> str:
        return f"{type(participant).__name__}:{id(participant)}:{int(ordinal)}"


__all__ = ["LifecycleDispatcher"]

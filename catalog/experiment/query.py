from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mlblack.capabilities.tracking import ExperimentRecord, SQLiteExperimentStore


@dataclass(frozen=True)
class ExperimentQuery:
    run_name: str | None = None
    event: str | None = None
    min_step: int | None = None
    max_step: int | None = None
    payload_contains: str = ""
    limit: int = 200

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "event": self.event,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "payload_contains": self.payload_contains,
            "limit": int(self.limit),
        }


@dataclass(frozen=True)
class ExperimentQueryResult:
    query: ExperimentQuery
    records: Sequence[ExperimentRecord]
    facets: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.as_dict(),
            "records": [record.as_dict() for record in self.records],
            "facets": {str(k): dict(v) for k, v in self.facets.items()},
        }


def query_experiments(path: str, query: ExperimentQuery | Mapping[str, Any] | None = None) -> ExperimentQueryResult:
    q = query if isinstance(query, ExperimentQuery) else ExperimentQuery(**dict(query or {}))
    records = list(SQLiteExperimentStore(path).list(run_name=q.run_name))
    if q.event is not None:
        records = [record for record in records if record.event == q.event]
    if q.min_step is not None:
        records = [record for record in records if record.step is not None and int(record.step) >= int(q.min_step)]
    if q.max_step is not None:
        records = [record for record in records if record.step is not None and int(record.step) <= int(q.max_step)]
    if q.payload_contains:
        needle = str(q.payload_contains)
        records = [record for record in records if needle in str(dict(record.payload))]
    records = records[: max(0, int(q.limit))]
    return ExperimentQueryResult(query=q, records=tuple(records), facets=_facets(records))


def _facets(records: Sequence[ExperimentRecord]) -> dict[str, dict[str, int]]:
    facets: dict[str, dict[str, int]] = {"run_name": {}, "event": {}}
    for record in records:
        facets["run_name"][record.run_name] = facets["run_name"].get(record.run_name, 0) + 1
        facets["event"][record.event] = facets["event"].get(record.event, 0) + 1
    return facets

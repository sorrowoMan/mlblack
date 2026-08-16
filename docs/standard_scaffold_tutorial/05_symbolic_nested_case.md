# 05. Symbolic Nested Case

Symbolic workload usually maps to:

- outer Case: structure/search semantics (often nsgablack)
- inner Case: parameter fitting / ML semantics (mlblack)

Both are standard Cases under one Project.

## 1) Contract

- each Case has one primary entry
- entry resolved by `.case kind`
- outer calls inner through standard payload, not private class coupling

## 2) Result Payload

Inner result should include:

- metrics/objectives
- canonical payload summary
- artifact refs
- resource audit fields

## 3) Resource Flow

Resource context flows:

```text
Project L0 -> outer Case -> child grant -> inner Case
```

No cross-layer self-authorization.

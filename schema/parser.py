from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.common.contracts import Cell, Sample, SampleDataset

from .spec import DatasetSchema, FeatureSpec, TargetSpec


class SchemaValidationError(ValueError):
    pass


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _validate_constraints_number(value: float, constraints: Mapping[str, Any], key: str) -> None:
    if "min" in constraints and value < float(constraints["min"]):
        raise SchemaValidationError(f"{key}: value {value} < min {constraints['min']}")
    if "max" in constraints and value > float(constraints["max"]):
        raise SchemaValidationError(f"{key}: value {value} > max {constraints['max']}")


def _validate_categorical(value: Any, *, vocab: Sequence[Any] | None, unknown: str, key: str) -> Any:
    if vocab is None:
        return value
    if value in vocab:
        return value
    if str(unknown).strip().lower() == "allow":
        return value
    raise SchemaValidationError(f"{key}: unknown category '{value}', vocab={list(vocab)}")


def _validate_numeric(value: Any, *, constraints: Mapping[str, Any], key: str) -> float:
    if not _is_numeric_scalar(value):
        raise SchemaValidationError(f"{key}: expected numeric scalar, got {type(value).__name__}")
    out = float(value)
    if not np.isfinite(out):
        raise SchemaValidationError(f"{key}: numeric value must be finite")
    _validate_constraints_number(out, constraints, key)
    return out


def _validate_integer(value: Any, *, constraints: Mapping[str, Any], key: str) -> int:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{key}: bool is not accepted for integer dtype")
    if not isinstance(value, (int, np.integer)):
        raise SchemaValidationError(f"{key}: expected integer, got {type(value).__name__}")
    out = int(value)
    _validate_constraints_number(float(out), constraints, key)
    return out


def _validate_boolean(value: Any, *, key: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SchemaValidationError(f"{key}: expected boolean, got {type(value).__name__}")
    return bool(value)


def _validate_text(value: Any, *, constraints: Mapping[str, Any], key: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{key}: expected text(str), got {type(value).__name__}")
    min_len = constraints.get("min_len")
    max_len = constraints.get("max_len")
    if min_len is not None and len(value) < int(min_len):
        raise SchemaValidationError(f"{key}: text length {len(value)} < min_len {min_len}")
    if max_len is not None and len(value) > int(max_len):
        raise SchemaValidationError(f"{key}: text length {len(value)} > max_len {max_len}")
    return value


def _validate_sequence(
    value: Any,
    *,
    key: str,
    item_dtype: str | None,
    constraints: Mapping[str, Any],
    vocab: Sequence[Any] | None,
    unknown: str,
) -> list[Any]:
    if isinstance(value, np.ndarray):
        seq = value.tolist()
    elif isinstance(value, (list, tuple)):
        seq = list(value)
    else:
        raise SchemaValidationError(f"{key}: expected sequence(list/tuple/ndarray), got {type(value).__name__}")

    min_len = constraints.get("min_len")
    max_len = constraints.get("max_len")
    if min_len is not None and len(seq) < int(min_len):
        raise SchemaValidationError(f"{key}: sequence length {len(seq)} < min_len {min_len}")
    if max_len is not None and len(seq) > int(max_len):
        raise SchemaValidationError(f"{key}: sequence length {len(seq)} > max_len {max_len}")

    if item_dtype is None:
        return seq

    out: list[Any] = []
    for idx, item in enumerate(seq):
        item_key = f"{key}[{idx}]"
        dt = str(item_dtype).strip().lower()
        if dt == "numeric":
            out.append(_validate_numeric(item, constraints={}, key=item_key))
        elif dt == "integer":
            out.append(_validate_integer(item, constraints={}, key=item_key))
        elif dt == "boolean":
            out.append(_validate_boolean(item, key=item_key))
        elif dt == "categorical":
            out.append(_validate_categorical(item, vocab=vocab, unknown=unknown, key=item_key))
        elif dt == "text":
            out.append(_validate_text(item, constraints={}, key=item_key))
        else:
            raise SchemaValidationError(f"{item_key}: unsupported item_dtype '{item_dtype}'")
    return out


def _validate_matrix(value: Any, *, key: str, constraints: Mapping[str, Any]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise SchemaValidationError(f"{key}: expected 2D matrix, got ndim={arr.ndim}")
    try:
        out = np.asarray(arr, dtype=float)
    except Exception as exc:
        raise SchemaValidationError(f"{key}: matrix must be numeric-convertible") from exc
    if not np.all(np.isfinite(out)):
        raise SchemaValidationError(f"{key}: matrix contains non-finite values")

    rows = constraints.get("rows")
    cols = constraints.get("cols")
    if rows is not None and int(out.shape[0]) != int(rows):
        raise SchemaValidationError(f"{key}: matrix rows {out.shape[0]} != expected {rows}")
    if cols is not None and int(out.shape[1]) != int(cols):
        raise SchemaValidationError(f"{key}: matrix cols {out.shape[1]} != expected {cols}")
    return out


def _validate_graph(value: Any, *, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{key}: graph expects mapping payload with keys like nodes/edges")
    if "nodes" not in value or "edges" not in value:
        raise SchemaValidationError(f"{key}: graph payload requires 'nodes' and 'edges'")
    return value


def _validate_by_dtype(
    value: Any,
    *,
    dtype: str,
    constraints: Mapping[str, Any],
    vocab: Sequence[Any] | None,
    unknown: str,
    item_dtype: str | None,
    key: str,
) -> Any:
    dt = str(dtype).strip().lower()

    if dt == "numeric":
        return _validate_numeric(value, constraints=constraints, key=key)
    if dt == "integer":
        return _validate_integer(value, constraints=constraints, key=key)
    if dt == "boolean":
        return _validate_boolean(value, key=key)
    if dt == "categorical":
        return _validate_categorical(value, vocab=vocab, unknown=unknown, key=key)
    if dt == "text":
        return _validate_text(value, constraints=constraints, key=key)
    if dt == "sequence":
        return _validate_sequence(
            value,
            key=key,
            item_dtype=item_dtype,
            constraints=constraints,
            vocab=vocab,
            unknown=unknown,
        )
    if dt == "matrix":
        return _validate_matrix(value, key=key, constraints=constraints)
    if dt == "graph":
        return _validate_graph(value, key=key)

    raise SchemaValidationError(f"{key}: unsupported dtype '{dtype}'")


def _validate_feature(row: Mapping[str, Any], spec: FeatureSpec) -> Cell:
    key = spec.key
    if key not in row:
        if spec.required:
            raise SchemaValidationError(f"feature '{key}' is missing")
        value = None
    else:
        value = row[key]

    if value is None:
        if spec.required:
            raise SchemaValidationError(f"feature '{key}' is None but required")
        payload = None
    else:
        payload = _validate_by_dtype(
            value,
            dtype=spec.dtype,
            constraints=spec.constraints,
            vocab=spec.vocab,
            unknown=spec.unknown,
            item_dtype=spec.item_dtype,
            key=key,
        )

    modality = str(spec.modality or spec.dtype).strip().lower()
    return Cell(name=key, payload=payload, modality=modality, labels={}, meta=dict(spec.meta))


def _validate_target(row: Mapping[str, Any], target: TargetSpec) -> Any:
    key = target.key
    if key not in row:
        if target.required:
            raise SchemaValidationError(f"target '{key}' is missing")
        return None

    value = row[key]
    if value is None and target.required:
        raise SchemaValidationError(f"target '{key}' is None but required")
    if value is None:
        return None

    return _validate_by_dtype(
        value,
        dtype=target.dtype,
        constraints=target.constraints,
        vocab=target.vocab,
        unknown="error",
        item_dtype=None,
        key=f"target.{key}",
    )


def _target_specs(schema: DatasetSchema) -> tuple[TargetSpec, ...]:
    out = tuple(schema.targets)
    if not out:
        raise SchemaValidationError("DatasetSchema.targets must not be empty")
    keys = [t.key for t in out]
    if len(set(keys)) != len(keys):
        raise SchemaValidationError("DatasetSchema.targets contains duplicated keys")
    return out


def _infer_target_names(value: Any, target_key: str) -> tuple[str, ...] | None:
    if value is None:
        return (target_key,)

    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None

    if arr.size <= 1:
        return (target_key,)
    return tuple(f"{target_key}[{i}]" for i in range(int(arr.size)))


def parse_row(row: Mapping[str, Any], schema: DatasetSchema, *, sample_id: str | None = None, row_index: int | None = None) -> Sample:
    targets = _target_specs(schema)

    if schema.strict:
        known = {f.key for f in schema.features}
        for t in targets:
            known.add(t.key)
        if schema.id_key:
            known.add(schema.id_key)
        extra = [k for k in row.keys() if k not in known]
        if extra:
            raise SchemaValidationError(f"unexpected keys in strict mode: {extra}")

    sid: str
    if sample_id is not None:
        sid = str(sample_id)
    elif schema.id_key and schema.id_key in row:
        sid = str(row[schema.id_key])
    elif row_index is not None:
        sid = str(row_index)
    else:
        sid = "sample"

    cells = {}
    for fs in schema.features:
        cell = _validate_feature(row, fs)
        cells[cell.name] = cell

    labels = {t.key: _validate_target(row, t) for t in targets}
    return Sample(sample_id=sid, cells=cells, labels=labels)


def parse_rows(rows: Sequence[Mapping[str, Any]], schema: DatasetSchema) -> SampleDataset:
    samples = [parse_row(row, schema, row_index=i) for i, row in enumerate(rows)]

    feature_keys = tuple(f.key for f in schema.features)
    targets = _target_specs(schema)
    default_target = targets[0]

    if samples:
        inferred_target_names = _infer_target_names(samples[0].labels.get(default_target.key), default_target.key)
    else:
        inferred_target_names = (default_target.key,)

    return SampleDataset(
        samples=samples,
        target_key=default_target.key,
        feature_cell_keys=feature_keys,
        target_names=inferred_target_names,
        description=schema.description,
    )

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .capabilities import TrainerCapabilities, coerce_trainer_capabilities
from .init import TrainingInit
from .signatures import (
    TrainingSignature,
    coerce_training_signature,
    signature_from_artifact,
    signature_from_state,
)


class TrainingCompatibilityError(ValueError):
    """Raised when a requested training mode is unsupported or underspecified."""

    def __init__(
        self,
        message: str,
        *,
        verdict: "CompatibilityVerdict | None" = None,
    ) -> None:
        super().__init__(str(message))
        self.verdict = verdict
        self.metadata = {} if verdict is None else dict(verdict.metadata)


@dataclass(frozen=True)
class CompatibilityVerdict:
    supported: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(str(v) for v in self.reasons))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))
        object.__setattr__(self, "metadata", dict(self.metadata))


_REQUIRED_SIGNATURE_FIELDS: dict[str, tuple[str, ...]] = {
    "resume": (
        "trainer_family",
        "schema_signature",
        "feature_signature",
        "target_signature",
        "objective_signature",
        "pipeline_signature",
        "numericizer_signature",
        "regime_signature",
        "symbolic_family_signature",
    ),
    "incremental": (
        "trainer_family",
        "schema_signature",
        "feature_signature",
        "target_signature",
        "objective_signature",
        "pipeline_signature",
        "numericizer_signature",
        "regime_signature",
        "symbolic_family_signature",
    ),
    "recalibrate": (
        "trainer_family",
        "schema_signature",
        "feature_signature",
        "target_signature",
        "objective_signature",
        "pipeline_signature",
        "numericizer_signature",
        "regime_signature",
        "symbolic_family_signature",
    ),
    "warm_start": (
        "trainer_family",
        "feature_signature",
        "target_signature",
        "pipeline_signature",
        "symbolic_family_signature",
    ),
}

_OPTIONAL_SIGNATURE_FIELDS: dict[str, tuple[str, ...]] = {
    "warm_start": (
        "schema_signature",
        "objective_signature",
        "numericizer_signature",
        "regime_signature",
        "symbolic_family_signature",
    ),
}


def _mapping_copy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if value is None else {str(k): v for k, v in dict(value).items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(dict(value).items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _symbolic_family_mapping(signature: TrainingSignature) -> dict[str, Any]:
    metadata = _mapping_copy(signature.metadata)
    raw = metadata.get("symbolic_family")
    if isinstance(raw, Mapping):
        return _mapping_copy(raw)
    return {}


def _extract_signature_contracts(signature: TrainingSignature) -> dict[str, dict[str, Any]]:
    symbolic_family = _symbolic_family_mapping(signature)
    rows = symbolic_family.get("search_family_signature_contracts", ())
    out: dict[str, dict[str, Any]] = {}
    for row in tuple(rows) if isinstance(rows, (list, tuple, set, frozenset)) else tuple():
        if not isinstance(row, Mapping):
            continue
        payload = _mapping_copy(row)
        key = str(payload.get("mechanism_key", "")).strip()
        if not key:
            continue
        out[key] = {str(k): _jsonable(v) for k, v in payload.items()}
    return out


def _extract_structure_signature_contracts(signature: TrainingSignature) -> dict[str, dict[str, Any]]:
    symbolic_family = _symbolic_family_mapping(signature)
    out: dict[str, dict[str, Any]] = {}
    structure_contracts = symbolic_family.get("structure_contracts")
    if isinstance(structure_contracts, Mapping):
        for _, value in sorted(dict(structure_contracts).items(), key=lambda item: str(item[0])):
            if not isinstance(value, Mapping):
                continue
            payload = _mapping_copy(value)
            key = str(payload.get("contract_key", "")).strip()
            if not key:
                continue
            out[key] = {str(k): _jsonable(v) for k, v in payload.items()}
    for singular_key in (
        "regime_discovery_contract",
        "basis_discovery_contract",
        "budgeted_symbolic_assembler_contract",
    ):
        row = symbolic_family.get(singular_key)
        if not isinstance(row, Mapping):
            continue
        payload = _mapping_copy(row)
        key = str(payload.get("contract_key", "")).strip()
        if not key or key in out:
            continue
        out[key] = {str(k): _jsonable(v) for k, v in payload.items()}
    return out


def _compare_symbolic_signature_contracts(
    *,
    current: TrainingSignature,
    parent: TrainingSignature,
    parent_label: str,
) -> dict[str, Any] | None:
    current_contracts = _extract_signature_contracts(current)
    parent_contracts = _extract_signature_contracts(parent)
    current_structure_contracts = _extract_structure_signature_contracts(current)
    parent_structure_contracts = _extract_structure_signature_contracts(parent)
    if not current_contracts and not parent_contracts and not current_structure_contracts and not parent_structure_contracts:
        return None

    changed: list[dict[str, Any]] = []
    for mechanism_key in sorted(set(current_contracts) | set(parent_contracts)):
        current_payload = current_contracts.get(mechanism_key)
        parent_payload = parent_contracts.get(mechanism_key)
        if current_payload is None:
            changed.append(
                {
                    "mechanism_key": str(mechanism_key),
                    "change_type": "missing_current",
                    "changed_fields": ("mechanism_key",),
                }
            )
            continue
        if parent_payload is None:
            changed.append(
                {
                    "mechanism_key": str(mechanism_key),
                    "change_type": "missing_parent",
                    "changed_fields": ("mechanism_key",),
                }
            )
            continue

        changed_fields = tuple(
            str(field_name)
            for field_name in sorted(set(current_payload) | set(parent_payload))
            if _jsonable(current_payload.get(field_name)) != _jsonable(parent_payload.get(field_name))
        )
        if changed_fields:
            changed.append(
                {
                    "mechanism_key": str(mechanism_key),
                    "change_type": "modified",
                    "changed_fields": changed_fields,
                    "current_contract": dict(current_payload),
                    "parent_contract": dict(parent_payload),
                }
            )

    changed_structure: list[dict[str, Any]] = []
    for contract_key in sorted(set(current_structure_contracts) | set(parent_structure_contracts)):
        current_payload = current_structure_contracts.get(contract_key)
        parent_payload = parent_structure_contracts.get(contract_key)
        if current_payload is None:
            changed_structure.append(
                {
                    "contract_key": str(contract_key),
                    "change_type": "missing_current",
                    "changed_fields": ("contract_key",),
                }
            )
            continue
        if parent_payload is None:
            changed_structure.append(
                {
                    "contract_key": str(contract_key),
                    "change_type": "missing_parent",
                    "changed_fields": ("contract_key",),
                }
            )
            continue

        changed_fields = tuple(
            str(field_name)
            for field_name in sorted(set(current_payload) | set(parent_payload))
            if _jsonable(current_payload.get(field_name)) != _jsonable(parent_payload.get(field_name))
        )
        if changed_fields:
            changed_structure.append(
                {
                    "contract_key": str(contract_key),
                    "change_type": "modified",
                    "changed_fields": changed_fields,
                    "current_contract": dict(current_payload),
                    "parent_contract": dict(parent_payload),
                }
            )

    if not changed and not changed_structure:
        return None

    labels: list[str] = []
    for row in changed:
        mechanism_key = str(row.get("mechanism_key", "unknown"))
        change_type = str(row.get("change_type", "modified"))
        changed_fields = tuple(str(v) for v in tuple(row.get("changed_fields", ())))
        if change_type == "modified":
            labels.append(f"{mechanism_key}[{','.join(changed_fields)}]")
        else:
            labels.append(f"{mechanism_key}[{change_type}]")
    for row in changed_structure:
        contract_key = str(row.get("contract_key", "unknown"))
        change_type = str(row.get("change_type", "modified"))
        changed_fields = tuple(str(v) for v in tuple(row.get("changed_fields", ())))
        if change_type == "modified":
            labels.append(f"{contract_key}[{','.join(changed_fields)}]")
        else:
            labels.append(f"{contract_key}[{change_type}]")

    return {
        "parent_label": str(parent_label),
        "current_signature": current.symbolic_family_signature,
        "parent_signature": parent.symbolic_family_signature,
        "current_mechanism_keys": tuple(sorted(current_contracts.keys())),
        "parent_mechanism_keys": tuple(sorted(parent_contracts.keys())),
        "changed_mechanisms": tuple(changed),
        "current_structure_contract_keys": tuple(sorted(current_structure_contracts.keys())),
        "parent_structure_contract_keys": tuple(sorted(parent_structure_contracts.keys())),
        "changed_structure_contracts": tuple(changed_structure),
        "changed_contracts": tuple([*changed, *changed_structure]),
        "message": "symbolic contract drift: " + "; ".join(labels),
    }


def _compare_signature(
    *,
    current: TrainingSignature,
    parent: TrainingSignature,
    parent_label: str,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...],
) -> tuple[list[str], list[str], dict[str, Any]]:
    reasons: list[str] = []
    warnings: list[str] = []
    comparison: dict[str, Any] = {
        "parent_label": str(parent_label),
        "required_fields": tuple(str(v) for v in required_fields),
        "optional_fields": tuple(str(v) for v in optional_fields),
        "mismatched_fields": [],
        "warning_fields": [],
    }

    for field_name in required_fields:
        current_value = getattr(current, field_name, None)
        parent_value = getattr(parent, field_name, None)
        if current_value is None and parent_value is None:
            continue
        if current_value is None:
            reasons.append(f"current task is missing required signature field '{field_name}'")
            continue
        if parent_value is None:
            reasons.append(f"{parent_label} is missing required signature field '{field_name}'")
            continue
        if str(current_value) != str(parent_value):
            reason = (
                f"{parent_label} mismatch on '{field_name}': current={current_value} parent={parent_value}"
            )
            if field_name == "symbolic_family_signature":
                drift = _compare_symbolic_signature_contracts(
                    current=current,
                    parent=parent,
                    parent_label=parent_label,
                )
                if drift is not None:
                    comparison["symbolic_family_signature_drift"] = drift
                    reason = f"{reason}; {str(drift.get('message', ''))}"
            reasons.append(reason)
            comparison["mismatched_fields"].append(str(field_name))

    for field_name in optional_fields:
        current_value = getattr(current, field_name, None)
        parent_value = getattr(parent, field_name, None)
        if current_value is None or parent_value is None:
            if current_value is not None and parent_value is None:
                warnings.append(f"{parent_label} is missing optional signature field '{field_name}'")
                comparison["warning_fields"].append(str(field_name))
            continue
        if str(current_value) != str(parent_value):
            reasons.append(
                f"{parent_label} mismatch on '{field_name}': current={current_value} parent={parent_value}"
            )
            comparison["mismatched_fields"].append(str(field_name))

    comparison["mismatched_fields"] = tuple(dict.fromkeys(comparison["mismatched_fields"]))
    comparison["warning_fields"] = tuple(dict.fromkeys(comparison["warning_fields"]))
    return reasons, warnings, comparison


def validate_training_setup(
    capabilities: TrainerCapabilities | Mapping[str, Any] | None,
    init: TrainingInit | None,
    *,
    current_signature: TrainingSignature | Mapping[str, Any] | None = None,
) -> CompatibilityVerdict:
    caps = coerce_trainer_capabilities(capabilities)
    training_init = init or TrainingInit()
    mode = str(training_init.mode)
    reasons: list[str] = []
    warnings: list[str] = []
    current_sig = coerce_training_signature(current_signature)

    if not caps.supports(mode):
        reasons.append(f"trainer does not declare support for mode '{mode}'")

    if mode == "resume" and training_init.parent_state is None:
        reasons.append("resume mode requires parent_state")
    elif mode == "warm_start" and training_init.parent_artifact is None and training_init.parent_state is None:
        reasons.append("warm_start mode requires parent_artifact or parent_state")
    elif mode == "incremental" and training_init.parent_artifact is None and training_init.parent_state is None:
        reasons.append("incremental mode requires parent_artifact or parent_state")
    elif mode == "recalibrate" and training_init.parent_artifact is None and training_init.parent_state is None:
        reasons.append("recalibrate mode requires parent_artifact or parent_state")

    required_fields = _REQUIRED_SIGNATURE_FIELDS.get(mode, ())
    optional_fields = _OPTIONAL_SIGNATURE_FIELDS.get(mode, ())
    if required_fields:
        comparison_rows: dict[str, Any] = {}
        symbolic_drift_rows: dict[str, Any] = {}
        if training_init.parent_state is not None:
            parent_sig = signature_from_state(training_init.parent_state)
            parent_reasons, parent_warnings, comparison = _compare_signature(
                current=current_sig,
                parent=parent_sig,
                parent_label="parent_state",
                required_fields=required_fields,
                optional_fields=optional_fields,
            )
            reasons.extend(parent_reasons)
            warnings.extend(parent_warnings)
            comparison_rows["parent_state"] = comparison
            drift = comparison.get("symbolic_family_signature_drift")
            if drift is not None:
                symbolic_drift_rows["parent_state"] = drift

        if training_init.parent_artifact is not None:
            parent_sig = signature_from_artifact(training_init.parent_artifact)
            parent_reasons, parent_warnings, comparison = _compare_signature(
                current=current_sig,
                parent=parent_sig,
                parent_label="parent_artifact",
                required_fields=required_fields,
                optional_fields=optional_fields,
            )
            reasons.extend(parent_reasons)
            warnings.extend(parent_warnings)
            comparison_rows["parent_artifact"] = comparison
            drift = comparison.get("symbolic_family_signature_drift")
            if drift is not None:
                symbolic_drift_rows["parent_artifact"] = drift
    else:
        comparison_rows = {}
        symbolic_drift_rows = {}

    return CompatibilityVerdict(
        supported=not reasons,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        metadata={
            "mode": mode,
            "capabilities": caps.as_dict(),
            "current_signature": current_sig.as_dict(),
            "required_signature_fields": required_fields,
            "optional_signature_fields": optional_fields,
            "signature_comparison": comparison_rows,
            "symbolic_family_signature_drift": symbolic_drift_rows,
        },
    )


def require_training_setup(
    capabilities: TrainerCapabilities | Mapping[str, Any] | None,
    init: TrainingInit | None,
    *,
    trainer_name: str,
    current_signature: TrainingSignature | Mapping[str, Any] | None = None,
) -> CompatibilityVerdict:
    verdict = validate_training_setup(capabilities, init, current_signature=current_signature)
    if not verdict.supported:
        raise TrainingCompatibilityError(
            f"{trainer_name} rejected training init for mode '{(init or TrainingInit()).mode}': "
            + "; ".join(verdict.reasons),
            verdict=verdict,
        )
    return verdict


__all__ = [
    "CompatibilityVerdict",
    "TrainingCompatibilityError",
    "require_training_setup",
    "validate_training_setup",
]

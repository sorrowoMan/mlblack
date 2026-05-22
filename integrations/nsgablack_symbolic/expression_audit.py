from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.models.symbolic import evaluate_expression_numpy, expression_to_string
from mlblack.models.symbolic_normalization import (
    expression_canonical_string,
    expression_equivalence_key,
    expression_family_signature,
    simplify_expression as engine_simplify_expression,
)


@dataclass(frozen=True)
class SymbolicExpressionAuditConfig:
    enable_simplification: bool = True
    enable_truth_contract_recovery: bool = True
    enable_equivalence_report: bool = True
    max_report_terms: int = 64
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolicExpressionAuditReport:
    simplified_expressions: Mapping[str, Any]
    simplification_trace: tuple[Mapping[str, Any], ...]
    truth_contract_recovery: Mapping[str, Any]
    equivalence_expression_handling: Mapping[str, Any]
    interference_feature_handling: Mapping[str, Any]
    periodic_equivalence_disambiguation: Mapping[str, Any]
    config: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "simplified_expressions": dict(self.simplified_expressions),
            "simplification_trace": [dict(row) for row in self.simplification_trace],
            "truth_contract_recovery": dict(self.truth_contract_recovery),
            "equivalence_expression_handling": dict(self.equivalence_expression_handling),
            "interference_feature_handling": dict(self.interference_feature_handling),
            "periodic_equivalence_disambiguation": dict(self.periodic_equivalence_disambiguation),
            "config": dict(self.config),
        }


class SymbolicExpressionAuditProducer:
    """Produces symbolic artifact audit sections from expressions and term metadata."""

    name = "symbolic_expression_audit_producer"
    context_requires = ("symbolic.expression_spec",)
    context_optional = ("data.X_train", "symbolic.function_pool", "basis.metrics", "artifact.report")
    context_provides = (
        "symbolic.simplification_trace",
        "symbolic.truth_contract_recovery",
        "symbolic.equivalence_report",
        "artifact.report",
    )
    context_mutates = ()
    context_cache = ()
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Builds simplification, equivalence, periodic/interference and truth-contract artifact sections."

    def __init__(self, config: SymbolicExpressionAuditConfig | None = None) -> None:
        self.config = config or SymbolicExpressionAuditConfig()

    def analyze(
        self,
        expressions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
        *,
        selected_terms: Sequence[Mapping[str, Any]] = tuple(),
        feature_names: Sequence[str] = tuple(),
        metadata: Mapping[str, Any] | None = None,
        X: np.ndarray | None = None,
    ) -> SymbolicExpressionAuditReport:
        expr_map = _coerce_expression_map(expressions)
        trace: list[Mapping[str, Any]] = []
        simplified: dict[str, Any] = {}
        for name, expr in expr_map.items():
            if bool(self.config.enable_simplification):
                simple = simplify_expression(expr, trace=trace, root=str(name))
            else:
                simple = dict(expr)
            simplified[str(name)] = {
                "expression": simple,
                "expression_string": expression_to_string(simple, feature_names=feature_names),
                "canonical_expression": expression_canonical_string(simple),
                "canonical_key": expression_equivalence_key(simple),
                "family_signature": expression_family_signature(simple, feature_names=tuple(feature_names)),
            }
        terms = tuple(dict(term) for term in tuple(selected_terms or ()))
        equivalence = (
            _equivalence_report(expr_map, simplified, terms, feature_names=feature_names, X=X, max_terms=int(self.config.max_report_terms))
            if bool(self.config.enable_equivalence_report)
            else {"status": "disabled"}
        )
        interference = _interference_report(terms, max_terms=int(self.config.max_report_terms))
        periodic = _periodic_report(terms, simplified, max_terms=int(self.config.max_report_terms))
        truth = (
            _truth_contract_recovery(metadata or {}, terms, simplified, feature_names=feature_names, max_terms=int(self.config.max_report_terms))
            if bool(self.config.enable_truth_contract_recovery)
            else {"status": "disabled"}
        )
        return SymbolicExpressionAuditReport(
            simplified_expressions=simplified,
            simplification_trace=tuple(trace[: max(0, int(self.config.max_report_terms))]),
            truth_contract_recovery=truth,
            equivalence_expression_handling=equivalence,
            interference_feature_handling=interference,
            periodic_equivalence_disambiguation=periodic,
            config={
                "enable_simplification": bool(self.config.enable_simplification),
                "enable_truth_contract_recovery": bool(self.config.enable_truth_contract_recovery),
                "enable_equivalence_report": bool(self.config.enable_equivalence_report),
                "max_report_terms": int(self.config.max_report_terms),
                "metadata": dict(self.config.metadata),
            },
        )

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "config": self.config.__dict__}


def simplify_expression(expr: Mapping[str, Any], *, trace: list[Mapping[str, Any]] | None = None, root: str = "$") -> dict[str, Any]:
    return engine_simplify_expression(expr, trace=trace, root=root)


def _coerce_expression_map(expressions: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if isinstance(expressions, Mapping):
        if "type" in expressions:
            return {"expr": dict(expressions)}
        return {str(key): dict(value) for key, value in dict(expressions).items()}
    out: dict[str, Mapping[str, Any]] = {}
    for idx, item in enumerate(tuple(expressions or ())):
        if "expr" in item:
            out[str(item.get("name", f"expr_{idx}"))] = dict(item["expr"])
        else:
            out[f"expr_{idx}"] = dict(item)
    return out


def _trace(root: str, rule: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(root),
        "rule": str(rule),
        "before": expression_to_string(before, precision=12),
        "after": expression_to_string(after, precision=12),
    }


def _const_value(expr: Mapping[str, Any]) -> float | None:
    return float(expr["value"]) if str(expr.get("type")) == "const" and expr.get("value") is not None else None


def _is_const(expr: Mapping[str, Any], value: float) -> bool:
    raw = _const_value(expr)
    return raw is not None and abs(float(raw) - float(value)) <= 1e-12


def _eval_unary_const(op: str, value: float) -> float:
    if op == "identity":
        return float(value)
    if op == "square":
        return float(value * value)
    if op == "sin":
        return float(np.sin(value))
    if op == "cos":
        return float(np.cos(value))
    if op == "tanh":
        return float(np.tanh(value))
    if op == "exp":
        return float(np.exp(np.clip(value, -30.0, 30.0)))
    if op == "log":
        return float(np.log(abs(value) + 1e-6))
    if op == "abs":
        return float(abs(value))
    if op == "sqrt":
        return float(np.sqrt(abs(value) + 1e-6))
    return float(value)


def _eval_binary_const(op: str, left: float, right: float) -> float:
    if op == "add":
        return float(left + right)
    if op == "sub":
        return float(left - right)
    if op == "mul":
        return float(left * right)
    if op == "div":
        denom = right if abs(right) > 1e-6 else (1e-6 if right >= 0.0 else -1e-6)
        return float(left / denom)
    return float(left)


def _algebraic_binary_simplify(op: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any] | None:
    if op == "add":
        if _is_const(left, 0.0):
            return dict(right)
        if _is_const(right, 0.0):
            return dict(left)
    if op == "sub" and _is_const(right, 0.0):
        return dict(left)
    if op == "mul":
        if _is_const(left, 0.0) or _is_const(right, 0.0):
            return {"type": "const", "value": 0.0}
        if _is_const(left, 1.0):
            return dict(right)
        if _is_const(right, 1.0):
            return dict(left)
    if op == "div":
        if _is_const(left, 0.0):
            return {"type": "const", "value": 0.0}
        if _is_const(right, 1.0):
            return dict(left)
    return None


def _equivalence_report(
    expr_map: Mapping[str, Mapping[str, Any]],
    simplified: Mapping[str, Any],
    terms: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str],
    X: np.ndarray | None,
    max_terms: int,
) -> dict[str, Any]:
    canonical: dict[str, dict[str, Any]] = {}
    for name, payload in dict(simplified).items():
        expr = dict(dict(payload).get("expression", {}) or {})
        key = expression_equivalence_key(expr) if expr else str(dict(payload).get("expression_string", ""))
        row = canonical.setdefault(
            key,
            {
                "canonical_key": key,
                "canonical_expression": str(dict(payload).get("expression_string", "")),
                "members": [],
            },
        )
        row["members"].append(str(name))
    duplicate_groups = [
        {"canonical_key": str(row["canonical_key"]), "canonical_expression": str(row["canonical_expression"]), "members": list(row["members"])}
        for row in canonical.values()
        if len(row["members"]) > 1
    ][: max(0, int(max_terms))]
    semantic: dict[str, list[str]] = {}
    for idx, term in enumerate(tuple(terms or ())):
        key = _term_semantic_key(term)
        semantic.setdefault(key, []).append(str(term.get("name", f"term_{idx}")))
    semantic_groups = [
        {"semantic_key": key, "members": members}
        for key, members in semantic.items()
        if len(members) > 1
    ][: max(0, int(max_terms))]
    family_groups = _family_equivalence_groups(simplified, max_terms=max_terms)
    phase_groups = _phase_equivalence_groups(simplified, max_terms=max_terms)
    value_groups = _value_equivalence_groups(expr_map, X, max_terms=max_terms) if X is not None else []
    return {
        "status": "reported",
        "canonical_duplicate_groups": duplicate_groups,
        "semantic_duplicate_groups": semantic_groups,
        "family_equivalence_groups": family_groups,
        "phase_equivalence_groups": phase_groups,
        "value_equivalence_groups": value_groups,
        "feature_names": list(feature_names),
    }


def _family_equivalence_groups(simplified: Mapping[str, Any], *, max_terms: int) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for name, payload in dict(simplified).items():
        signature = dict(dict(payload).get("family_signature", {}) or {})
        family = str(signature.get("family", ""))
        features = ",".join(str(v) for v in tuple(signature.get("features", ()) or ()))
        key = f"{family}|{features}"
        groups.setdefault(key, []).append(str(name))
    return [
        {"family_key": key, "members": members}
        for key, members in groups.items()
        if key.strip("|") and len(members) > 1
    ][: max(0, int(max_terms))]


def _phase_equivalence_groups(simplified: Mapping[str, Any], *, max_terms: int) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for name, payload in dict(simplified).items():
        signature = dict(dict(payload).get("family_signature", {}) or {})
        phase_key = str(signature.get("phase_equivalence_key", ""))
        if phase_key:
            groups.setdefault(phase_key, []).append(str(name))
    return [
        {"phase_equivalence_key": key, "members": members}
        for key, members in groups.items()
        if len(members) > 1
    ][: max(0, int(max_terms))]


def _value_equivalence_groups(expr_map: Mapping[str, Mapping[str, Any]], X: np.ndarray | None, *, max_terms: int) -> list[dict[str, Any]]:
    if X is None:
        return []
    names = list(dict(expr_map).keys())
    values: list[np.ndarray] = []
    for name in names:
        try:
            values.append(np.asarray(evaluate_expression_numpy(expr_map[name], X), dtype=float).reshape(-1))
        except Exception:
            values.append(np.full((np.asarray(X).shape[0],), np.nan, dtype=float))
    rows: list[dict[str, Any]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            left = values[i]
            right = values[j]
            if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                continue
            denom = float(max(1e-12, np.linalg.norm(left), np.linalg.norm(right)))
            distance = float(np.linalg.norm(left - right) / denom)
            if distance <= 1e-8:
                rows.append({"left": names[i], "right": names[j], "relative_l2": distance})
                if len(rows) >= int(max_terms):
                    return rows
    return rows


def _term_semantic_key(term: Mapping[str, Any]) -> str:
    family = str(term.get("family", term.get("activation_family", "")) or "")
    features = ",".join(str(int(v)) for v in tuple(term.get("features", ()) or ()))
    return f"{family}|{features}"


def _interference_report(terms: Sequence[Mapping[str, Any]], *, max_terms: int) -> dict[str, Any]:
    feature_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for idx, term in enumerate(tuple(terms or ())):
        family = str(term.get("family", term.get("activation_family", "")) or "")
        family_counts[family] = int(family_counts.get(family, 0) + 1)
        for feature in tuple(term.get("features", ()) or ()):
            key = str(int(feature))
            feature_counts[key] = int(feature_counts.get(key, 0) + 1)
        rows.append(
            {
                "term_index": int(idx),
                "name": str(term.get("name", f"term_{idx}")),
                "family": family,
                "features": [int(v) for v in tuple(term.get("features", ()) or ())],
            }
        )
    repeated = {key: value for key, value in feature_counts.items() if int(value) > 1}
    return {
        "status": "reported",
        "feature_reuse": repeated,
        "feature_counts": feature_counts,
        "family_counts": family_counts,
        "terms": rows[: max(0, int(max_terms))],
    }


def _periodic_report(
    terms: Sequence[Mapping[str, Any]],
    simplified: Mapping[str, Any],
    *,
    max_terms: int,
) -> dict[str, Any]:
    periodic_rows: list[dict[str, Any]] = []
    for idx, term in enumerate(tuple(terms or ())):
        family = str(term.get("family", term.get("activation_family", "")) or "").lower()
        name = str(term.get("name", f"term_{idx}"))
        expr_text = expression_to_string(dict(term.get("expr", {}) or {}), precision=12) if isinstance(term.get("expr"), Mapping) else name
        signature = (
            expression_family_signature(dict(term.get("expr", {}) or {}))
            if isinstance(term.get("expr"), Mapping)
            else {}
        )
        if any(token in family or token in name.lower() or token in expr_text.lower() for token in ("sin", "cos", "tan", "periodic", "trig")):
            periodic_rows.append(
                {
                    "term_index": int(idx),
                    "name": name,
                    "family": family,
                    "features": [int(v) for v in tuple(term.get("features", ()) or ())],
                    "phase_equivalence_key": str(signature.get("phase_equivalence_key", "")),
                    "phase_equivalence_policy": "scored",
                }
            )
    simplified_periodic = [
        {
            "name": str(name),
            "expression": str(dict(payload).get("expression_string", "")),
            "phase_equivalence_key": str(dict(dict(payload).get("family_signature", {}) or {}).get("phase_equivalence_key", "")),
        }
        for name, payload in dict(simplified).items()
        if str(dict(dict(payload).get("family_signature", {}) or {}).get("phase_equivalence_key", ""))
        or re.search(r"\b(sin|cos|tan)\(", str(dict(payload).get("expression_string", "")).lower())
    ]
    return {
        "status": "reported",
        "periodic_term_count": int(len(periodic_rows) + len(simplified_periodic)),
        "periodic_terms": periodic_rows[: max(0, int(max_terms))],
        "periodic_expressions": simplified_periodic[: max(0, int(max_terms))],
        "phase_equivalence_policy": "scored",
    }


def _truth_contract_recovery(
    metadata: Mapping[str, Any],
    terms: Sequence[Mapping[str, Any]],
    simplified: Mapping[str, Any],
    *,
    feature_names: Sequence[str],
    max_terms: int,
) -> dict[str, Any]:
    contracts = _truth_contracts_from_metadata(metadata)
    if not contracts:
        return {"status": "not_recorded", "source": "metadata.truth_contracts", "contract_count": 0}
    term_rows = [_truth_term_row(term, feature_names=feature_names) for term in tuple(terms or ())]
    expr_rows = [
        {
            "name": str(name),
            "expression": str(dict(payload).get("expression_string", "")),
            "canonical_key": str(dict(payload).get("canonical_key", "")),
            "canonical_expression": str(dict(payload).get("canonical_expression", "")),
            "family_signature": dict(dict(payload).get("family_signature", {}) or {}),
            "family": _family_from_expression_string(str(dict(payload).get("expression_string", ""))),
            "features": _features_from_expression_string(str(dict(payload).get("expression_string", "")), feature_names),
        }
        for name, payload in dict(simplified).items()
    ]
    matches: list[dict[str, Any]] = []
    for contract in contracts:
        spec = _parse_contract(str(contract), feature_names=feature_names)
        term_evaluations = [_contract_match_status(spec, row) for row in term_rows]
        expr_evaluations = [_contract_match_status(spec, row) for row in expr_rows]
        matched_terms = [row for row, status in zip(term_rows, term_evaluations) if bool(status["matched"])]
        matched_exprs = [row for row, status in zip(expr_rows, expr_evaluations) if bool(status["matched"])]
        best_status = _best_contract_status((*term_evaluations, *expr_evaluations))
        matches.append(
            {
                "contract": str(contract),
                "spec": spec,
                "matched": bool(matched_terms or matched_exprs),
                "match_status": best_status,
                "term_matches": matched_terms[: max(0, int(max_terms))],
                "expression_matches": matched_exprs[: max(0, int(max_terms))],
            }
        )
    matched_count = sum(1 for row in matches if bool(row["matched"]))
    exact_count = sum(1 for row in matches if bool(dict(row.get("match_status", {})).get("exact")))
    family_count = sum(1 for row in matches if bool(dict(row.get("match_status", {})).get("family")))
    phase_count = sum(1 for row in matches if bool(dict(row.get("match_status", {})).get("phase_equivalent")))
    family_recovery = {
        "status": "reported",
        "family_matched_contract_count": int(family_count),
        "phase_equivalent_contract_count": int(phase_count),
        "family_recovery_score": float(family_count / float(max(1, len(contracts)))),
        "phase_equivalence_recovery_score": float(phase_count / float(max(1, len(contracts)))),
        "families_requested": sorted({str(dict(row.get("spec", {})).get("family", "")) for row in matches}),
    }
    return {
        "status": "reported",
        "source": "metadata.truth_contracts",
        "contract_count": int(len(contracts)),
        "matched_contract_count": int(matched_count),
        "exact_contract_match_count": int(exact_count),
        "family_matched_contract_count": int(family_count),
        "phase_equivalent_contract_count": int(phase_count),
        "exact_term_recovery_score": float(exact_count / float(max(1, len(contracts)))),
        "family_recovery_score": float(family_count / float(max(1, len(contracts)))),
        "phase_equivalence_recovery_score": float(phase_count / float(max(1, len(contracts)))),
        "family_recovery": family_recovery,
        "matches": matches[: max(0, int(max_terms))],
    }


def _truth_contracts_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    meta = dict(metadata or {})
    candidates = (
        meta.get("truth_contracts"),
        dict(meta.get("truth_formula", {}) or {}).get("contracts") if isinstance(meta.get("truth_formula"), Mapping) else None,
        dict(dict(meta.get("data_metadata", {}) or {}).get("truth_formula", {}) or {}).get("contracts")
        if isinstance(meta.get("data_metadata"), Mapping)
        else None,
    )
    out: list[str] = []
    for value in candidates:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            out.extend(str(v) for v in value)
    return tuple(str(v).strip() for v in out if str(v).strip())


def _truth_term_row(term: Mapping[str, Any], *, feature_names: Sequence[str]) -> dict[str, Any]:
    features = [int(v) for v in tuple(term.get("features", ()) or ())]
    expr = dict(term.get("expr", {}) or {}) if isinstance(term.get("expr"), Mapping) else {}
    signature = expression_family_signature(expr, feature_names=tuple(feature_names)) if expr else {}
    return {
        "name": str(term.get("name", "")),
        "family": str(term.get("family", term.get("activation_family", "")) or "").lower(),
        "family_signature": signature,
        "canonical_key": str(signature.get("canonical_key", "")),
        "canonical_expression": str(signature.get("canonical_expression", "")),
        "features": [str(tuple(feature_names)[idx]) if 0 <= idx < len(tuple(feature_names)) else f"x{idx}" for idx in features],
        "feature_indices": features,
    }


def _parse_contract(contract: str, *, feature_names: Sequence[str]) -> dict[str, Any]:
    text = str(contract).strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text)
    if match:
        family = match.group(1).strip().lower()
        features = [part.strip() for part in match.group(2).split(",") if part.strip()]
        return {"family": family, "features": features, "raw": text}
    return {"family": "linear_feature", "features": [text], "raw": text}


def _contract_match_status(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    exact = _matches_contract(spec, row)
    family = exact or _matches_family(spec, row)
    phase = _matches_phase_equivalent(spec, row)
    return {
        "matched": bool(exact or family or phase),
        "exact": bool(exact),
        "family": bool(family),
        "phase_equivalent": bool(phase),
    }


def _best_contract_status(statuses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(statuses or ())
    return {
        "matched": any(bool(row.get("matched", False)) for row in rows),
        "exact": any(bool(row.get("exact", False)) for row in rows),
        "family": any(bool(row.get("family", False)) for row in rows),
        "phase_equivalent": any(bool(row.get("phase_equivalent", False)) for row in rows),
    }


def _matches_contract(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    family = str(spec.get("family", "")).lower()
    want_features = {str(v).lower() for v in tuple(spec.get("features", ()) or ())}
    row_features = {str(v).lower() for v in tuple(row.get("features", ()) or ())}
    row_family = str(row.get("family", "")).lower()
    expr = str(row.get("expression", row.get("name", ""))).lower()
    if want_features and not want_features.issubset(row_features | set(re.findall(r"\bx\d+\b", expr))):
        return False
    if family == "linear_feature":
        return bool(want_features & (row_features | {expr}))
    if family in row_family or family in expr:
        return True
    aliases = {"safe_ratio": ("ratio", "div"), "square": ("square", "^2"), "sine": ("sin",), "cosine": ("cos",)}
    return any(alias in row_family or alias in expr for alias in aliases.get(family, ()))


def _matches_family(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    want_features = {str(v).lower() for v in tuple(spec.get("features", ()) or ())}
    row_features = {str(v).lower() for v in tuple(row.get("features", ()) or ())}
    signature = dict(row.get("family_signature", {}) or {})
    row_features |= {str(v).lower() for v in tuple(signature.get("features", ()) or ())}
    if want_features and not want_features.issubset(row_features):
        return False
    family = str(spec.get("family", "")).lower()
    row_family = str(row.get("family", "")).lower()
    sig_family = str(signature.get("family", "")).lower()
    sig_subfamily = str(signature.get("subfamily", "")).lower()
    aliases = {
        "sin": {"trig", "sin", "sine"},
        "sine": {"trig", "sin", "sine"},
        "cos": {"trig", "cos", "cosine"},
        "cosine": {"trig", "cos", "cosine"},
        "square": {"power", "square", "trig_power"},
        "safe_ratio": {"ratio", "division"},
        "ratio": {"ratio", "division"},
        "exp": {"exponential", "exp", "log_exp_chain"},
        "log": {"logarithmic", "log", "exp_log_chain"},
    }
    wanted = aliases.get(family, {family})
    observed = {row_family, sig_family, sig_subfamily}
    return bool(wanted & observed)


def _matches_phase_equivalent(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    family = str(spec.get("family", "")).lower()
    if family not in {"sin", "sine", "cos", "cosine", "trig", "periodic"}:
        return False
    signature = dict(row.get("family_signature", {}) or {})
    if not str(signature.get("phase_equivalence_key", "")):
        return False
    want_features = {str(v).lower() for v in tuple(spec.get("features", ()) or ())}
    row_features = {str(v).lower() for v in tuple(row.get("features", ()) or ())}
    row_features |= {str(v).lower() for v in tuple(signature.get("features", ()) or ())}
    return not want_features or want_features.issubset(row_features)


def _family_from_expression_string(expr: str) -> str:
    text = str(expr).lower()
    for family in ("sin", "cos", "tanh", "exp", "log", "sqrt", "abs"):
        if f"{family}(" in text:
            return family
    if "/" in text:
        return "ratio"
    if "^2" in text:
        return "square"
    return "expression"


def _features_from_expression_string(expr: str, feature_names: Sequence[str]) -> list[str]:
    text = str(expr)
    features = set(re.findall(r"\bx\d+\b", text))
    for name in tuple(feature_names):
        if str(name) in text:
            features.add(str(name))
    return sorted(features)


__all__ = [
    "SymbolicExpressionAuditConfig",
    "SymbolicExpressionAuditProducer",
    "SymbolicExpressionAuditReport",
    "simplify_expression",
]

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

TRUTH_CONTRACT_RECOVERY_MIN_NORMALIZED_WEIGHT = 0.01


def normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def normalized_feature_tuple(values: Any) -> tuple[str, ...]:
    normalized = [normalized_text(value) for value in tuple(values or ())]
    return tuple(sorted(value for value in normalized if value))


def first_list_mapping_value(mapping: Any) -> list[dict[str, Any]]:
    if isinstance(mapping, Sequence) and not isinstance(mapping, (str, bytes, bytearray)):
        if all(isinstance(row, Mapping) for row in mapping):
            return [dict(row) for row in mapping]
    for value in dict(mapping or {}).values():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if all(isinstance(row, Mapping) for row in value):
                return [dict(row) for row in value]
        if isinstance(value, Mapping):
            nested = first_list_mapping_value(value)
            if nested:
                return nested
    return []


def expr_looks_like_safe_ratio(expr: str, *, numerator: str, denominator: str) -> bool:
    normalized = normalized_text(expr)
    if not normalized:
        return False
    if "/safe(" in normalized:
        return numerator in normalized and denominator in normalized
    return (
        "/" in normalized
        and numerator in normalized
        and denominator in normalized
        and ("abs(" in normalized or "safe" in normalized)
    )


def expr_looks_like_piecewise_hinge(expr: str, *, feature_name: str) -> bool:
    normalized = normalized_text(expr)
    if not normalized or feature_name not in normalized:
        return False
    if "relu(" in normalized or "hinge" in normalized or "piecewise" in normalized:
        return True
    # Standard hinge reparameterization: 0.5 * (z + abs(z)).
    return ("abs(" in normalized) and ("0.5" in normalized)


def term_row_view(row: Mapping[str, Any]) -> dict[str, Any]:
    expr = str(
        row.get("expression_named")
        or row.get("expression_raw")
        or row.get("expression")
        or row.get("expr")
        or ""
    )
    return {
        "term_name": str(row.get("term_name") or row.get("name") or ""),
        "name": normalized_text(row.get("term_name") or row.get("name") or ""),
        "expr": normalized_text(expr),
        "expression": expr,
        "features": normalized_feature_tuple(row.get("feature_names", ()) or row.get("features", ())),
        "semantic_family": normalized_text(row.get("semantic_family")),
        "semantic_signature": normalized_text(row.get("semantic_signature")),
        "uses_piecewise_gate": bool(row.get("uses_piecewise_gate")),
        "coefficient": row.get("coefficient"),
        "abs_coefficient": row.get("abs_coefficient"),
        "normalized_weight": row.get("normalized_weight"),
        "node_count": row.get("node_count"),
        "selection_channel": normalized_text(row.get("selection_channel")),
        "source_object_key": normalized_text(row.get("source_object_key")),
        "source_support_key": normalized_text(row.get("source_support_key")),
        "chart_signature": normalized_text(row.get("chart_signature")),
        "structural_channel": normalized_text(row.get("structural_channel")),
        "support_expansion_tagged": bool(row.get("support_expansion_tagged")),
        "canonical_trunk_tagged": bool(row.get("canonical_trunk_tagged")),
        "same_source_surrogate_tagged": bool(row.get("same_source_surrogate_tagged")),
        "support_expansion_candidate": bool(row.get("support_expansion_candidate")),
        "canonical_trunk_candidate": bool(row.get("canonical_trunk_candidate")),
        "same_source_surrogate_candidate": bool(row.get("same_source_surrogate_candidate")),
    }


def truth_basis_rows_from_basis_structure(basis_structure: Mapping[str, Any]) -> list[dict[str, Any]]:
    recorded = dict(dict(dict(basis_structure or {}).get("basis_semantics", {}) or {}).get("recorded", {}) or {})
    basis_terms = recorded.get("basis_terms")
    if isinstance(basis_terms, Sequence) and not isinstance(basis_terms, (str, bytes, bytearray)):
        return [dict(row) for row in basis_terms if isinstance(row, Mapping)]
    return first_list_mapping_value(dict(basis_structure or {}).get("global_basis"))


def _base_contract_spec(
    *,
    contract: str,
    family: str,
    match_kind: str,
    features: Sequence[str],
    ordered_features: Sequence[str] | None = None,
    numerator_features: Sequence[str] | None = None,
    denominator_features: Sequence[str] | None = None,
    match_mode: str = "exact",
    accepted_families: Sequence[str] = (),
    expected_sign: str | None = None,
) -> dict[str, Any]:
    return {
        "contract": str(contract),
        "family": normalized_text(family),
        "match_kind": normalized_text(match_kind or family),
        "features": normalized_feature_tuple(features),
        "ordered_features": tuple(normalized_text(value) for value in tuple(ordered_features or features) if normalized_text(value)),
        "numerator_features": normalized_feature_tuple(numerator_features or ()),
        "denominator_features": normalized_feature_tuple(denominator_features or ()),
        "match_mode": normalized_text(match_mode or "exact") or "exact",
        "accepted_families": tuple(
            normalized_text(value)
            for value in tuple(accepted_families or ())
            if normalized_text(value)
        ),
        "expected_sign": None if expected_sign is None else str(expected_sign),
    }


def _contract_spec_from_mapping(value: Mapping[str, Any], *, default_match_mode: str) -> dict[str, Any]:
    features = tuple(value.get("features", ()) or value.get("feature_names", ()) or ())
    contract = str(value.get("contract") or value.get("label") or "")
    family = str(value.get("family") or value.get("match_kind") or "unknown")
    match_kind = str(value.get("match_kind") or family or "unknown")
    if not contract:
        feature_text = ",".join(str(item) for item in tuple(features))
        contract = f"{family}({feature_text})" if feature_text else family
    return _base_contract_spec(
        contract=contract,
        family=family,
        match_kind=match_kind,
        features=features,
        ordered_features=tuple(value.get("ordered_features", ()) or features),
        numerator_features=tuple(value.get("numerator_features", ()) or ()),
        denominator_features=tuple(value.get("denominator_features", ()) or ()),
        match_mode=str(value.get("match_mode") or default_match_mode or "exact"),
        accepted_families=tuple(value.get("accepted_families", ()) or ()),
        expected_sign=None if value.get("expected_sign") is None else str(value.get("expected_sign")),
    )


def _contract_spec_from_string(contract: str, *, default_match_mode: str) -> dict[str, Any] | None:
    text = str(contract or "").strip()
    normalized = normalized_text(text)
    if not normalized:
        return None
    args: list[str]
    if normalized.startswith("safe_ratio(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        numerator = str(args[0]) if len(args) >= 1 else ""
        denominator = str(args[1]) if len(args) >= 2 else ""
        return _base_contract_spec(
            contract=text,
            family="safe_ratio",
            match_kind="safe_ratio",
            features=(numerator, denominator),
            ordered_features=(numerator, denominator),
            numerator_features=(numerator,),
            denominator_features=(denominator,),
            match_mode=default_match_mode,
        )
    if normalized.startswith("product_ratio(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        numerator_features = tuple(args[:-1])
        denominator_feature = tuple(args[-1:]) if args else tuple()
        return _base_contract_spec(
            contract=text,
            family="product_ratio",
            match_kind="product_ratio",
            features=args,
            ordered_features=args,
            numerator_features=numerator_features,
            denominator_features=denominator_feature,
            match_mode=default_match_mode,
        )
    if normalized.startswith("exp_ratio(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        numerator = str(args[0]) if len(args) >= 1 else ""
        denominator = str(args[1]) if len(args) >= 2 else ""
        return _base_contract_spec(
            contract=text,
            family="exp_ratio",
            match_kind="exp_ratio",
            features=(numerator, denominator),
            ordered_features=(numerator, denominator),
            numerator_features=(numerator,),
            denominator_features=(denominator,),
            match_mode=default_match_mode,
        )
    if normalized.startswith("piecewise_hinge(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="piecewise_hinge",
            match_kind="piecewise_hinge",
            features=args[:1],
            ordered_features=args[:1],
            match_mode=default_match_mode,
        )
    if normalized.startswith("periodic_phase_equivalent(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="periodic_unary",
            match_kind="periodic_phase_equivalent",
            features=args[:1],
            ordered_features=args[:1],
            match_mode="phase_equivalent",
            accepted_families=("sin", "cos"),
        )
    if normalized.startswith("periodic_family(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="periodic_unary",
            match_kind="periodic_family",
            features=args[:1],
            ordered_features=args[:1],
            match_mode="family",
            accepted_families=("sin", "cos"),
        )
    if normalized.startswith("ratio_family(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="ratio_or_reciprocal",
            match_kind="ratio_family",
            features=args[:2],
            ordered_features=args[:2],
            numerator_features=args[:1],
            denominator_features=args[1:2],
            match_mode="family",
        )
    if normalized.startswith("product_ratio_family(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="ratio_or_reciprocal",
            match_kind="product_ratio_family",
            features=args,
            ordered_features=args,
            numerator_features=args[:-1],
            denominator_features=args[-1:],
            match_mode="family",
        )
    if normalized.startswith("exp_ratio_family(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="single_feature_transform",
            match_kind="exp_ratio_family",
            features=args[:2],
            ordered_features=args[:2],
            numerator_features=args[:1],
            denominator_features=args[1:2],
            match_mode="family",
        )
    if normalized.startswith("piecewise_gate_family(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="piecewise_gate",
            match_kind="piecewise_gate_family",
            features=args[:1],
            ordered_features=args[:1],
            match_mode="family",
        )
    if normalized.startswith("linear_feature_family(") and normalized.endswith(")"):
        args = [part.strip() for part in text[text.find("(") + 1 : -1].split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family="linear_feature",
            match_kind="linear_feature",
            features=args[:1],
            ordered_features=args[:1],
            match_mode="family",
        )
    unary_match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)\(([^()]*)\)", text)
    if unary_match:
        family = normalized_text(unary_match.group(1))
        args = [part.strip() for part in unary_match.group(2).split(",") if part.strip()]
        return _base_contract_spec(
            contract=text,
            family=family,
            match_kind=family,
            features=args,
            ordered_features=args,
            match_mode=default_match_mode,
        )
    return _base_contract_spec(
        contract=text,
        family="linear_feature",
        match_kind="linear_feature",
        features=(text,),
        ordered_features=(text,),
        match_mode=default_match_mode,
    )


def truth_contract_specs(
    contracts: Sequence[str | Mapping[str, Any]],
    *,
    default_match_mode: str = "exact",
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for contract in tuple(contracts or ()):
        spec: dict[str, Any] | None
        if isinstance(contract, Mapping):
            spec = _contract_spec_from_mapping(contract, default_match_mode=default_match_mode)
        else:
            spec = _contract_spec_from_string(str(contract), default_match_mode=default_match_mode)
        if spec is not None:
            specs.append(spec)
    return specs


def _features_match(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    expected = tuple(spec.get("features", ()))
    if not expected:
        return True
    return tuple(row.get("features", ())) == expected


def _match_ratio_like(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not _features_match(spec, row):
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    numerator = str(tuple(spec.get("numerator_features", ()) or ("",))[0])
    denominator = str(tuple(spec.get("denominator_features", ()) or ("",))[0])
    if numerator and denominator:
        if expr_looks_like_safe_ratio(expr, numerator=numerator, denominator=denominator):
            return True
        if expr_looks_like_safe_ratio(name, numerator=numerator, denominator=denominator):
            return True
    return (
        ("ratio" in semantic_family)
        or ("binary:div" in semantic_signature)
        or ("/" in expr)
        or ("/" in name)
    )


def _match_product_ratio_like(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not _features_match(spec, row):
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    numerator_features = tuple(spec.get("numerator_features", ()))
    denominator_features = tuple(spec.get("denominator_features", ()))
    denominator = str(denominator_features[0] if denominator_features else "")
    if all(feature in expr for feature in numerator_features) and denominator in expr and "/" in expr:
        return True
    if all(feature in name for feature in numerator_features) and denominator in name and "/" in name:
        return True
    return (
        (("ratio" in semantic_family) or ("binary:div" in semantic_signature))
        and all(feature in tuple(row.get("features", ())) for feature in numerator_features)
        and (not denominator or denominator in tuple(row.get("features", ())))
    )


def _text_has_ordered_ratio(text: str, *, numerator: str, denominator: str) -> bool:
    normalized = re.sub(r"\s+", "", normalized_text(text))
    numerator = normalized_text(numerator)
    denominator = normalized_text(denominator)
    if not normalized or not numerator or not denominator or "/" not in normalized:
        return False
    # Direction matters for exact exp-ratio contracts: exp(-A/B) is not
    # interchangeable with exp(-B/A), even though they share one source set.
    pattern = re.escape(numerator) + r"[^/]{0,160}/[^,;+\-*]{0,200}" + re.escape(denominator)
    return bool(re.search(pattern, normalized))


def _match_exp_ratio_like(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not _features_match(spec, row):
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    numerator = str(tuple(spec.get("numerator_features", ()) or ("",))[0])
    denominator = str(tuple(spec.get("denominator_features", ()) or ("",))[0])
    match_mode = str(spec.get("match_mode") or "").strip().lower()
    if "exp(" in expr and _text_has_ordered_ratio(expr, numerator=numerator, denominator=denominator):
        return True
    if "exp(" in name and _text_has_ordered_ratio(name, numerator=numerator, denominator=denominator):
        return True
    if match_mode == "family":
        if "exp(" in expr and numerator in expr and denominator in expr and "/" in expr:
            return True
        if "exp(" in name and numerator in name and denominator in name and "/" in name:
            return True
        return ("unary:exp" in semantic_signature) and ("binary:div" in semantic_signature)
    return False


def _match_piecewise_like(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not _features_match(spec, row):
        return False
    feature_name = str(tuple(spec.get("ordered_features", ()) or ("",))[0])
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    return bool(
        row.get("uses_piecewise_gate")
        or ("piecewise" in semantic_family)
        or ("hinge" in name)
        or ("relu(" in expr)
        or expr_looks_like_piecewise_hinge(expr, feature_name=feature_name)
        or expr_looks_like_piecewise_hinge(name, feature_name=feature_name)
    )


def _match_periodic_like(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not _features_match(spec, row):
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    accepted = tuple(spec.get("accepted_families", ()) or ("sin", "cos"))
    for family in accepted:
        if (f"{family}(" in expr) or (f"{family}(" in name) or (f"unary:{family}" in semantic_signature):
            return True
    return "periodic" in semantic_family


def _match_linear_feature(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    if not _features_match(spec, row):
        return False
    feature_name = str(tuple(spec.get("ordered_features", ()) or ("",))[0])
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_family = str(row.get("semantic_family") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    node_count = row.get("node_count")
    return bool(
        expr == feature_name
        or name == feature_name
        or semantic_family == "linear_feature"
        or semantic_signature.startswith("feature:")
        or node_count == 1
    )


def matches_truth_contract(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    match_kind = str(spec.get("match_kind") or spec.get("family") or "")
    if match_kind in {"safe_ratio", "ratio_family"}:
        return _match_ratio_like(spec, row)
    if match_kind in {"product_ratio", "product_ratio_family"}:
        return _match_product_ratio_like(spec, row)
    if match_kind in {"exp_ratio", "exp_ratio_family"}:
        return _match_exp_ratio_like(spec, row)
    if match_kind in {"piecewise_hinge", "piecewise_gate_family"}:
        return _match_piecewise_like(spec, row)
    if match_kind in {"periodic_phase_equivalent", "periodic_family"}:
        return _match_periodic_like(spec, row)
    if match_kind == "linear_feature":
        return _match_linear_feature(spec, row)
    if not _features_match(spec, row):
        return False
    expr = str(row.get("expr") or "")
    name = str(row.get("name") or "")
    semantic_signature = str(row.get("semantic_signature") or "")
    family = str(spec.get("family") or "")
    return (f"{family}(" in expr) or (f"{family}(" in name) or (f"unary:{family}" in semantic_signature)


def row_sign_matches(expected_sign: str | None, row: Mapping[str, Any]) -> bool:
    coefficient = row.get("coefficient")
    if expected_sign is None or coefficient is None:
        return True
    try:
        numeric = float(coefficient)
    except (TypeError, ValueError):
        return True
    if str(expected_sign) == "positive":
        return numeric > 0.0
    if str(expected_sign) == "negative":
        return numeric < 0.0
    return True


def row_is_materially_active(row: Mapping[str, Any], *, min_normalized_weight: float) -> bool:
    weight = row.get("normalized_weight")
    if weight is not None:
        try:
            return float(weight) >= float(min_normalized_weight)
        except (TypeError, ValueError):
            pass
    coefficient = row.get("abs_coefficient")
    if coefficient is None:
        coefficient = row.get("coefficient")
    try:
        return abs(float(coefficient)) > 1e-10
    except (TypeError, ValueError):
        return False


def build_contract_match_summary(
    *,
    contracts: Sequence[str | Mapping[str, Any]],
    basis_rows: Sequence[Mapping[str, Any]],
    active_term_rows: Sequence[Mapping[str, Any]],
    min_normalized_weight: float,
    default_match_mode: str = "exact",
) -> dict[str, Any]:
    specs = truth_contract_specs(contracts, default_match_mode=default_match_mode)
    contract_matches: list[dict[str, Any]] = []
    matched_basis_count = 0
    matched_term_count = 0
    for spec in specs:
        basis_matches = [row for row in basis_rows if matches_truth_contract(spec, row)]
        active_matches = [
            row
            for row in active_term_rows
            if matches_truth_contract(spec, row)
            and row_is_materially_active(row, min_normalized_weight=min_normalized_weight)
            and row_sign_matches(spec.get("expected_sign"), row)
        ]
        basis_hit = bool(basis_matches)
        term_hit = bool(active_matches)
        outer_chart_hit = bool(basis_matches)
        inner_realization_only = bool(term_hit and not basis_hit)
        matched_basis_count += int(basis_hit)
        matched_term_count += int(term_hit)
        contract_matches.append(
            {
                "truth_term": str(spec.get("contract") or ""),
                "truth_family": str(spec.get("family") or ""),
                "truth_features": [str(value) for value in tuple(spec.get("features", ()))],
                "match_mode": str(spec.get("match_mode") or default_match_mode),
                "expected_sign": spec.get("expected_sign"),
                "basis_hit": basis_hit,
                "outer_chart_hit": outer_chart_hit,
                "term_recovered": term_hit,
                "inner_realization_only": inner_realization_only,
                "matched_basis_terms": [str(row.get("term_name") or row.get("expression") or "") for row in basis_matches],
                "matched_basis_expressions": [str(row.get("expression") or "") for row in basis_matches],
                "matched_basis_selection_channels": [str(row.get("selection_channel") or "") for row in basis_matches],
                "matched_basis_chart_signatures": [str(row.get("chart_signature") or "") for row in basis_matches],
                "matched_expression_terms": [str(row.get("expression") or "") for row in active_matches],
                "matched_expression_coefficients": [row.get("coefficient") for row in active_matches],
            }
        )
    truth_count = int(len(specs))
    return {
        "contract_count": truth_count,
        "matched_basis_count": int(matched_basis_count),
        "matched_term_count": int(matched_term_count),
        "basis_hit_score": (None if truth_count <= 0 else float(matched_basis_count) / float(truth_count)),
        "term_recovery_score": (None if truth_count <= 0 else float(matched_term_count) / float(truth_count)),
        "matches": contract_matches,
    }


def build_truth_contract_recovery(
    *,
    truth_formula: Mapping[str, Any],
    basis_rows: Sequence[Mapping[str, Any]],
    active_term_rows: Sequence[Mapping[str, Any]],
    min_normalized_weight: float = TRUTH_CONTRACT_RECOVERY_MIN_NORMALIZED_WEIGHT,
    source: str = "data_metadata.truth_formula",
) -> dict[str, Any]:
    strict_contract = tuple(truth_formula.get("strict_contract", ()) or truth_formula.get("basis_contract", ()) or ())
    phase_equivalent_contract = tuple(truth_formula.get("phase_equivalent_contract", ()) or ())
    family_level_contract = tuple(truth_formula.get("family_level_contract", ()) or ())
    if not strict_contract:
        return {
            "status": "not_recorded",
            "source": "not_recorded",
            "truth_formula_expression": None,
            "truth_basis_count": 0,
            "matched_truth_basis_count": 0,
            "matched_truth_term_count": 0,
            "exact_basis_hit_score": None,
            "outer_chart_hit_score": None,
            "exact_term_recovery_score": None,
            "inner_realization_hit_score": None,
            "inner_realization_only_score": None,
            "exact_term_min_normalized_weight": float(min_normalized_weight),
            "truth_basis_matches": [],
            "phase_equivalent_contract_count": 0,
            "phase_equivalent_basis_hit_score": None,
            "phase_equivalent_term_recovery_score": None,
            "phase_equivalent_matches": [],
            "family_level_contract_count": 0,
            "family_level_basis_hit_score": None,
            "family_level_term_recovery_score": None,
            "family_level_matches": [],
        }

    strict_summary = build_contract_match_summary(
        contracts=strict_contract,
        basis_rows=basis_rows,
        active_term_rows=active_term_rows,
        min_normalized_weight=min_normalized_weight,
        default_match_mode="exact",
    )
    phase_summary = build_contract_match_summary(
        contracts=phase_equivalent_contract,
        basis_rows=basis_rows,
        active_term_rows=active_term_rows,
        min_normalized_weight=min_normalized_weight,
        default_match_mode="phase_equivalent",
    ) if phase_equivalent_contract else {
        "contract_count": 0,
        "basis_hit_score": None,
        "term_recovery_score": None,
        "matches": [],
    }
    family_summary = build_contract_match_summary(
        contracts=family_level_contract,
        basis_rows=basis_rows,
        active_term_rows=active_term_rows,
        min_normalized_weight=min_normalized_weight,
        default_match_mode="family",
    ) if family_level_contract else {
        "contract_count": 0,
        "basis_hit_score": None,
        "term_recovery_score": None,
        "matches": [],
    }
    return {
        "status": "reported",
        "source": f"{source}.strict_contract",
        "truth_formula_expression": (
            None if truth_formula.get("expression") is None else str(truth_formula.get("expression"))
        ),
        "truth_basis_count": int(strict_summary.get("contract_count", 0) or 0),
        "matched_truth_basis_count": int(strict_summary.get("matched_basis_count", 0) or 0),
        "matched_truth_term_count": int(strict_summary.get("matched_term_count", 0) or 0),
        "exact_basis_hit_score": strict_summary.get("basis_hit_score"),
        "outer_chart_hit_score": strict_summary.get("basis_hit_score"),
        "exact_term_recovery_score": strict_summary.get("term_recovery_score"),
        "inner_realization_hit_score": strict_summary.get("term_recovery_score"),
        "inner_realization_only_score": (
            None
            if strict_summary.get("basis_hit_score") is None or strict_summary.get("term_recovery_score") is None
            else max(
                0.0,
                float(strict_summary.get("term_recovery_score") or 0.0)
                - float(strict_summary.get("basis_hit_score") or 0.0),
            )
        ),
        "exact_term_min_normalized_weight": float(min_normalized_weight),
        "truth_basis_matches": list(strict_summary.get("matches", []) or []),
        "phase_equivalent_contract_count": int(phase_summary.get("contract_count", 0) or 0),
        "phase_equivalent_basis_hit_score": phase_summary.get("basis_hit_score"),
        "phase_equivalent_term_recovery_score": phase_summary.get("term_recovery_score"),
        "phase_equivalent_matches": list(phase_summary.get("matches", []) or []),
        "family_level_contract_count": int(family_summary.get("contract_count", 0) or 0),
        "family_level_basis_hit_score": family_summary.get("basis_hit_score"),
        "family_level_term_recovery_score": family_summary.get("term_recovery_score"),
        "family_level_matches": list(family_summary.get("matches", []) or []),
    }

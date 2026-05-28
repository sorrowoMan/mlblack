from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract, ContractMixin
from mlblack.pipeline.data_views import NumericDataView
from .plan import NumericFeatureColumn, NumericizationPlan


@dataclass
class DefaultNumericizer(ContractMixin):
    name = "default_numericizer"
    context_requires = ("data.raw_rows", "data.schema")
    context_optional = ("data.target",)
    context_provides = ("data.numeric_view", "pipeline.feature_space")
    context_mutates = ()
    context_cache = ("data.schema",)
    requires_metrics = ()
    metrics_fallback = "strict"
    context_notes = "Fits raw rows against DatasetSchema and returns a NumericDataView."
    contract = ComponentContract(
        name=name,
        requires=("data.raw_rows", "data.schema"),
        optional=("data.target",),
        provides=("data.numeric_view", "pipeline.feature_space"),
        cache=("data.schema",),
        supports_batch=True,
        supports_resume=True,
        metadata={"layer": "pipeline", "component": "numericizer"},
    )

    plan: NumericizationPlan

    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "DefaultNumericizer":
        # Late-bind categorical vocab when schema did not provide one.
        columns: list[NumericFeatureColumn] = []
        for column in self.plan.columns:
            if column.kind != "categorical_pending":
                columns.append(column)
                continue
            values = sorted({str(row.get(column.source_key, "")) for row in rows})
            for value in values:
                columns.append(NumericFeatureColumn(name=f"{column.source_key}={value}", source_key=column.source_key, kind="onehot", metadata={"value": value}))
        if tuple(columns) != tuple(self.plan.columns):
            self.plan = NumericizationPlan(schema=self.plan.schema, columns=tuple(columns), target=self.plan.target, metadata={**dict(self.plan.metadata), "late_bound_vocab": True})
        return self

    def transform(self, rows: Sequence[Mapping[str, Any]]) -> NumericDataView:
        row_list = [dict(row) for row in rows]
        if not row_list:
            raise ValueError("DefaultNumericizer.transform requires at least one row")
        X = np.asarray([[self._value_for_column(row, column) for column in self.plan.columns] for row in row_list], dtype=float)
        y = np.asarray([float(row[self.plan.target.key]) for row in row_list], dtype=float)
        return NumericDataView(
            X_train=X,
            y_train=y,
            feature_names=self.plan.feature_names,
            target_name=self.plan.target.key,
            metadata={"numericization_plan": self.plan.as_dict()},
        )

    def fit_transform(self, rows: Sequence[Mapping[str, Any]]) -> NumericDataView:
        return self.fit(rows).transform(rows)

    def _value_for_column(self, row: Mapping[str, Any], column: NumericFeatureColumn) -> float:
        if column.kind == "onehot":
            return 1.0 if str(row.get(column.source_key, "")) == str(column.metadata.get("value", "")) else 0.0
        if column.kind == "boolean":
            return 1.0 if bool(row.get(column.source_key, False)) else 0.0
        return float(row.get(column.source_key, 0.0))


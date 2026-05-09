from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from core.common.contracts import ProcessedDataset
from core.orchestration.workflow import TrainDataBundle
from examples.path_defaults import default_work_ci_csv


@dataclass
class WorkCiIntervalReader:
    """Reader for the work-CI interval dataset table."""

    csv_path: str = default_work_ci_csv()
    target_col: str = "ci"
    date_col: str = "date"
    test_fold_col: str = "test_fold_10"
    extra_drop_cols: Sequence[str] = ()

    def read(self) -> TrainDataBundle:
        df = pd.read_csv(self.csv_path)
        if df.empty:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        if self.target_col not in df.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in CSV")

        if self.test_fold_col not in df.columns:
            raise ValueError(f"test_fold_col '{self.test_fold_col}' not found in CSV")

        fold_cols = [c for c in df.columns if c.startswith("test_fold_")]

        drop_cols = set(fold_cols)
        drop_cols.add(self.target_col)
        if self.date_col in df.columns:
            drop_cols.add(self.date_col)
        drop_cols.update(str(c) for c in self.extra_drop_cols)

        feature_cols = [c for c in df.columns if c not in drop_cols]
        if not feature_cols:
            raise ValueError("No feature columns selected")

        X_df = df[feature_cols].copy()
        Y_df = df[[self.target_col]].copy()

        # Strong numeric requirement for this processed-table reader.
        for c in feature_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        Y_df[self.target_col] = pd.to_numeric(Y_df[self.target_col], errors="coerce")

        if X_df.isna().any().any() or Y_df.isna().any().any():
            na_cols = [c for c in X_df.columns if X_df[c].isna().any()]
            if Y_df[self.target_col].isna().any():
                na_cols.append(self.target_col)
            raise ValueError(f"Found NaN after numeric conversion. Columns: {na_cols}")

        X_all = X_df.to_numpy(dtype=float)
        y_all = Y_df.to_numpy(dtype=float)

        test_mask = pd.to_numeric(df[self.test_fold_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        if int(np.sum(test_mask)) == 0:
            raise ValueError(f"No test rows where {self.test_fold_col} == 1")

        train_mask = ~test_mask

        train_ds = ProcessedDataset(
            X_train=np.asarray(X_all[train_mask], dtype=float),
            y_train=np.asarray(y_all[train_mask], dtype=float),
            feature_names=tuple(feature_cols),
            target_names=(self.target_col,),
            metadata={
                "source": self.csv_path,
                "split": "train",
                "split_rule": f"{self.test_fold_col}==0",
            },
        )

        test_ds = ProcessedDataset(
            X_train=np.asarray(X_all[test_mask], dtype=float),
            y_train=np.asarray(y_all[test_mask], dtype=float),
            feature_names=tuple(feature_cols),
            target_names=(self.target_col,),
            metadata={
                "source": self.csv_path,
                "split": "test",
                "split_rule": f"{self.test_fold_col}==1",
            },
        )

        return TrainDataBundle(
            train=train_ds,
            test=test_ds,
            metadata={
                "reader": "WorkCiIntervalReader",
                "n_total": int(len(df)),
                "n_train": int(np.sum(train_mask)),
                "n_test": int(np.sum(test_mask)),
                "target_col": self.target_col,
                "feature_cols": tuple(feature_cols),
                "test_fold_col": self.test_fold_col,
            },
        )

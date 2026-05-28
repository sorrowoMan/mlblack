"""Cloud and external data-source pipeline components.

These are DataPipelineComponent subclasses that read from remote systems
(S3, Hive, JDBC, etc.) and produce NumericDataView objects for downstream
pipeline stages.  Each source is the *first* node in a DataPipeline chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from mlblack.core.contracts import ComponentContract
from mlblack.pipeline.base import DataPipelineComponent
from mlblack.pipeline.data_views import NumericDataView


@dataclass(frozen=True)
class S3ParquetSource(DataPipelineComponent):
    """Read a Parquet dataset from S3 or an S3-compatible store (MinIO, etc.).

    The source lists all ``.parquet`` files under ``bucket/prefix``, reads them
    via ``pyarrow``, and returns a single ``NumericDataView`` from the combined
    table.  Column names are carried forward as ``feature_names`` / ``target_name``.

    Optional dependencies: ``pyarrow``, ``s3fs``.
    """

    name = "s3_parquet_source"
    bucket: str = ""
    prefix: str = ""
    endpoint_url: str = ""               # leave empty for AWS S3; set for MinIO etc.
    access_key: str = ""
    secret_key: str = ""
    region: str = ""
    target_column: str = ""              # name of the label column; last column if empty
    timeout: int = 30
    context_requires = ()                # source — no upstream data dependency
    context_optional = ("trainer.context",)
    context_provides = ("data.numeric_view",)
    context_mutates = ("pipeline.component_state",)
    context_notes = (
        "Reads Parquet from S3/MinIO and produces a NumericDataView. "
        "Optional deps: pyarrow, s3fs."
    )
    contract = ComponentContract(
        name=name,
        requires=(),
        optional=("trainer.context",),
        provides=("data.numeric_view",),
        mutates=("pipeline.component_state",),
        supports_batch=True,
        metadata={"layer": "pipeline", "source": "s3_parquet"},
    )

    def fit(self, data, context=None):
        return {}

    def transform(self, data, state=None, context=None):
        try:
            import pyarrow.parquet as pq
            import s3fs
        except ImportError as exc:
            raise ImportError(
                "S3ParquetSource requires pyarrow + s3fs.  pip install pyarrow s3fs"
            ) from exc

        fs_kwargs: dict = {"key": str(self.access_key), "secret": str(self.secret_key)}
        if self.endpoint_url:
            fs_kwargs["endpoint_url"] = str(self.endpoint_url)
        if self.region:
            fs_kwargs["region_name"] = str(self.region)
        fs = s3fs.S3FileSystem(**fs_kwargs)

        glob_pattern = f"{self.bucket.strip('/')}/{self.prefix.strip('/')}/*.parquet"
        files = sorted(fs.glob(glob_pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files at s3://{glob_pattern}")

        table = pq.ParquetDataset(files, filesystem=fs).read()
        df = table.to_pandas()
        columns = [str(c) for c in table.column_names]
        target = str(self.target_column) if self.target_column else columns[-1]
        feature_cols = [c for c in columns if c != target]

        X = np.asarray(df[feature_cols], dtype=float)
        y = np.asarray(df[target], dtype=float)
        metadata = {
            "source": f"s3://{glob_pattern}",
            "file_count": len(files),
            "total_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }
        return NumericDataView(
            X_train=X,
            y_train=y,
            feature_names=tuple(feature_cols),
            target_name=target,
            metadata=metadata,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bucket": str(self.bucket),
            "prefix": str(self.prefix),
            "endpoint_url": str(self.endpoint_url),
            "target_column": str(self.target_column),
        }


@dataclass(frozen=True)
class HiveQuerySource(DataPipelineComponent):
    """Run a SQL query against HiveServer2 and materialise the result set.

    The query result is converted to a ``NumericDataView``.  The first N-1 columns
    become features; the last column becomes the target (overridable via
    ``target_column``).

    Optional dependencies: ``pyhive``, ``sasl`` (or ``thrift`` transport).
    """

    name = "hive_query_source"
    host: str = "localhost"
    port: int = 10000
    database: str = "default"
    query: str = ""
    auth: str = "NONE"                   # NONE | KERBEROS | LDAP | CUSTOM
    username: str = ""
    password: str = ""
    target_column: str = ""
    timeout: int = 120
    context_requires = ()                # source — no upstream data dependency
    context_optional = ("trainer.context",)
    context_provides = ("data.numeric_view",)
    context_mutates = ("pipeline.component_state",)
    context_notes = (
        "Executes a SQL query against HiveServer2 and materialises the result. "
        "Optional deps: pyhive."
    )
    contract = ComponentContract(
        name=name,
        requires=(),
        optional=("trainer.context",),
        provides=("data.numeric_view",),
        mutates=("pipeline.component_state",),
        supports_batch=True,
        metadata={"layer": "pipeline", "source": "hive"},
    )

    def fit(self, data, context=None):
        return {}

    def transform(self, data, state=None, context=None):
        try:
            from pyhive import hive
        except ImportError as exc:
            raise ImportError(
                "HiveQuerySource requires pyhive.  pip install pyhive thrift sasl"
            ) from exc

        if not self.query.strip():
            raise ValueError("HiveQuerySource requires a non-empty query.")

        conn = hive.connect(
            host=str(self.host),
            port=int(self.port),
            database=str(self.database),
            auth=str(self.auth),
            username=str(self.username),
            password=str(self.password),
        )
        try:
            cursor = conn.cursor()
            cursor.execute(str(self.query))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        finally:
            conn.close()

        if not rows:
            raise RuntimeError("Hive query returned zero rows.")

        arr = np.asarray(rows, dtype=float)
        target = str(self.target_column) if self.target_column else (columns[-1] if columns else "")
        col_indices = {name: idx for idx, name in enumerate(columns)}
        feature_cols = [c for c in columns if c != target]

        X = arr[:, [col_indices[c] for c in feature_cols]]
        y = arr[:, col_indices[target]] if target in col_indices else arr[:, -1]
        metadata = {
            "source": f"hive://{self.host}:{self.port}/{self.database}",
            "total_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }
        return NumericDataView(
            X_train=X,
            y_train=y,
            feature_names=tuple(feature_cols),
            target_name=target,
            metadata=metadata,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": str(self.host),
            "port": int(self.port),
            "database": str(self.database),
            "target_column": str(self.target_column),
        }


@dataclass(frozen=True)
class JDBCQuerySource(DataPipelineComponent):
    """Run a SQL query against any JDBC data source and materialise the result.

    Uses ``JayDeBeApi`` under the hood.  Column behaviour follows ``HiveQuerySource``.

    Optional dependencies: ``jaydebeapi`` + a JDBC driver jar.
    """

    name = "jdbc_query_source"
    jdbc_url: str = ""
    driver_class: str = ""
    driver_jar: str = ""
    query: str = ""
    username: str = ""
    password: str = ""
    target_column: str = ""
    timeout: int = 120
    context_requires = ()
    context_optional = ("trainer.context",)
    context_provides = ("data.numeric_view",)
    context_mutates = ("pipeline.component_state",)
    context_notes = (
        "Generic JDBC query source.  Optional deps: jaydebeapi + JDBC driver."
    )
    contract = ComponentContract(
        name=name,
        requires=(),
        optional=("trainer.context",),
        provides=("data.numeric_view",),
        mutates=("pipeline.component_state",),
        supports_batch=True,
        metadata={"layer": "pipeline", "source": "jdbc"},
    )

    def fit(self, data, context=None):
        return {}

    def transform(self, data, state=None, context=None):
        try:
            import jaydebeapi
        except ImportError as exc:
            raise ImportError(
                "JDBCQuerySource requires jaydebeapi.  pip install jaydebeapi"
            ) from exc

        if not self.jdbc_url or not self.query.strip():
            raise ValueError("JDBCQuerySource requires jdbc_url and query.")

        conn = jaydebeapi.connect(
            str(self.driver_class),
            str(self.jdbc_url),
            [str(self.username), str(self.password)],
            str(self.driver_jar),
        )
        try:
            cursor = conn.cursor()
            cursor.execute(str(self.query))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
        finally:
            conn.close()

        if not rows:
            raise RuntimeError("JDBC query returned zero rows.")

        arr = np.asarray(rows, dtype=float)
        target = str(self.target_column) if self.target_column else (columns[-1] if columns else "")
        col_indices = {name: idx for idx, name in enumerate(columns)}
        feature_cols = [c for c in columns if c != target]

        X = arr[:, [col_indices[c] for c in feature_cols]]
        y = arr[:, col_indices[target]] if target in col_indices else arr[:, -1]
        metadata = {
            "source": str(self.jdbc_url),
            "total_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        }
        return NumericDataView(
            X_train=X,
            y_train=y,
            feature_names=tuple(feature_cols),
            target_name=target,
            metadata=metadata,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "jdbc_url": str(self.jdbc_url),
            "driver_class": str(self.driver_class),
            "target_column": str(self.target_column),
        }

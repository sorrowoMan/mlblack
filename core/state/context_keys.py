from __future__ import annotations

# run metadata
RUN_NAME = "run.name"
RUN_STAGE = "run.stage"
RUN_STARTED_AT = "run.started_at_utc"
RUN_FINISHED_AT = "run.finished_at_utc"

# data refs
FLOW_SPEC_REF = "flow.spec_ref"
BUNDLE_REF = "data.bundle_ref"
PROCESSED_REF = "data.processed_ref"
NUMERICIZER_REF = "data.numericizer_ref"

# model refs
MODEL_SPEC_REF = "model.spec_ref"
MODEL_PROCESSED_REF = "model.processed_ref"

# train/eval refs
TRAINER_REF = "train.trainer_ref"
ARTIFACT_REF = "train.artifact_ref"
TRAINER_STATE_REF = "train.trainer_state_ref"
EVAL_SPLITS = "eval.splits"
METRICS_REF = "eval.metrics_ref"

# report/result refs
REPORT_REF = "flow.report_ref"
RESULT_REF = "flow.result_ref"


__all__ = [
    "RUN_NAME",
    "RUN_STAGE",
    "RUN_STARTED_AT",
    "RUN_FINISHED_AT",
    "FLOW_SPEC_REF",
    "BUNDLE_REF",
    "PROCESSED_REF",
    "NUMERICIZER_REF",
    "MODEL_SPEC_REF",
    "MODEL_PROCESSED_REF",
    "TRAINER_REF",
    "ARTIFACT_REF",
    "TRAINER_STATE_REF",
    "EVAL_SPLITS",
    "METRICS_REF",
    "REPORT_REF",
    "RESULT_REF",
]

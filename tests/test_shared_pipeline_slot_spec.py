from __future__ import annotations

from blackbase.kernel import PipelineSlotSpec as SharedPipelineSlotSpec
from mlblack.pipeline.slot_kernel import PipelineSlotSpec, PipelineSpec


def test_mlblack_exports_the_complete_shared_pipeline_spec() -> None:
    assert PipelineSlotSpec is SharedPipelineSlotSpec
    spec = PipelineSpec.from_value(
        {
            "key": "ml",
            "slots": [
                {
                    "slot": "transform",
                    "operators": ["normalize"],
                    "stages": [[0, "normalize"]],
                    "index_key": "branch_index",
                    "timeout_seconds": 2.5,
                    "cancel_on_error": False,
                }
            ],
        }
    )

    slot = spec.slot_specs()[0]
    assert slot.stages == ((0, "normalize"),)
    assert slot.index_key == "branch_index"
    assert slot.timeout_seconds == 2.5
    assert slot.cancel_on_error is False

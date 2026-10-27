def test_upstream_annotation_metadata_survives_wrapper(monkeypatch):
    from modal_dojo.frameworks.miles.modal_helpers import (
        convert_hf_to_torch_dist as wrapper,
    )

    monkeypatch.setattr(wrapper, "_start_memory_monitor", lambda: None)
    monkeypatch.setattr(
        wrapper,
        "_load_upstream_source",
        lambda: (
            """
from typing import Annotated
from pydantic import BaseModel
class ConversionArgs(BaseModel):
    hf_checkpoint: Annotated[str, "required-cli-option"]
assert ConversionArgs.model_fields["hf_checkpoint"].metadata == ["required-cli-option"]
"""
        ),
    )
    wrapper.main()

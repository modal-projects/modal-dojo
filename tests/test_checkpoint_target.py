from modal_dojo.common.checkpoint import Checkpoint, CheckpointType
from modal_dojo.common.checkpoint_target import CheckpointTarget, ServingSpec
from modal_dojo.common.models import ModelConfig


def test_from_checkpoint_builds_volume_reference():
    checkpoint = Checkpoint(
        checkpoint_type=CheckpointType.megatron,
        name="iter_0000040",
        path="/checkpoints/run-1/iter_0000040",
        timestamp=0.0,
        checkpoints_volume_name="dojo-checkpoints",
        checkpoints_mount_path="/checkpoints",
    )
    target = CheckpointTarget.from_checkpoint(
        checkpoint,
        ModelConfig(model_name="Qwen/Qwen3-4B"),
        serving=ServingSpec(gpu="H100", min_containers=2),
    )

    assert target.to_request() == {
        "checkpoint": {
            "volume_name": "dojo-checkpoints",
            "path": "run-1/iter_0000040",
            "base_model": "Qwen/Qwen3-4B",
            "format": "megatron",
            "environment": None,
            "serving": {
                "gpu": "H100",
                "min_containers": 2,
                "max_containers": None,
                "max_inputs_per_container": None,
            },
        }
    }

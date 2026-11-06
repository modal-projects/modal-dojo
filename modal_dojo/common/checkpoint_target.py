"""POC (option D): a training checkpoint as a first-class Modal Evals target.

Not for merge. Today every caller (Dojo, the midtrain scheduler, Eval Zoo)
converts, serves, scales and tears down a checkpoint itself before Modal Evals
can reach it. This POC sketches the target type Modal Evals would accept
instead, so callers only pass a reference to weights on a Volume:

    target = CheckpointTarget.from_checkpoint(run.latest_checkpoint(), model)
    run_evaluation("terminal-bench-2-1", target, wait=True)

Proposed server-side shape (``flash_eval_api.proto``)::

    message CheckpointTarget {
      string volume_name = 1;        // Volume that holds the weights
      string path = 2;               // path inside the Volume
      string base_model = 3;         // HF repo of the architecture / tokenizer
      CheckpointFormat format = 4;   // HF | MEGATRON | LORA_ADAPTER
      string environment = 5;        // where the Volume lives
      ServingSpec serving = 6;       // optional; unset = auto
    }
    message ServingSpec {
      string gpu = 1;
      uint32 min_containers = 2;     // pre-warm from the job's trial_concurrency
      uint32 max_containers = 3;
      uint32 max_inputs_per_container = 4;
    }

Modal Evals would own convert -> deploy -> autoscale on trial concurrency ->
retry 429s -> tear down. Until that exists, :meth:`CheckpointTarget.serve` is
a client-side shim with the same contract, built on Dojo's ``Endpoint``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from modal_dojo.common.checkpoint import Checkpoint
    from modal_dojo.common.endpoint import Endpoint
    from modal_dojo.common.models import ModelConfig

CheckpointFormat = Literal["hf", "megatron", "lora_adapter"]


@dataclass(frozen=True)
class ServingSpec:
    """How Modal Evals should serve the checkpoint; every field optional."""

    gpu: str | None = None
    min_containers: int | None = None
    max_containers: int | None = None
    max_inputs_per_container: int | None = None


@dataclass(frozen=True)
class CheckpointTarget:
    """Weights on a Modal Volume that Modal Evals serves for the job.

    Attributes:
        volume_name: Modal Volume holding the checkpoint.
        path: Checkpoint directory inside the Volume.
        base_model: Hugging Face repo of the architecture and tokenizer.
        format: Weight layout; ``megatron`` is converted to HF before serving.
        environment: Modal environment of the Volume.
        serving: Serving overrides; ``None`` lets Modal Evals choose.
    """

    volume_name: str
    path: str
    base_model: str
    format: CheckpointFormat = "megatron"
    environment: str | None = None
    serving: ServingSpec | None = None
    # Client-side shim only: the Dojo model config used to serve the weights.
    _model: Any = field(default=None, compare=False, repr=False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        model: ModelConfig,
        *,
        environment: str | None = None,
        serving: ServingSpec | None = None,
    ) -> CheckpointTarget:
        from modal_dojo.common.checkpoint import CheckpointType

        return cls(
            volume_name=checkpoint.checkpoints_volume_name,
            path=checkpoint.path_relative_to_volume,
            base_model=model.model_name,
            format="hf"
            if checkpoint.checkpoint_type is CheckpointType.hf
            else "megatron",
            environment=environment,
            serving=serving,
            _model=model,
        )

    def to_request(self) -> dict[str, Any]:
        """The proposed ``CheckpointTarget`` proto, as JSON."""
        payload = asdict(self)
        payload.pop("_model")
        return {"checkpoint": payload}

    @contextlib.contextmanager
    def serve(self) -> Iterator[Endpoint]:
        """Shim for what Modal Evals would do server-side: serve the weights
        for the length of the block, then stop them."""
        from modal_dojo.common.checkpoint import Checkpoint, CheckpointType
        from modal_dojo.common.endpoint import Endpoint

        if self._model is None:
            raise ValueError(
                "The client-side shim needs a Dojo ModelConfig; build the "
                "target with CheckpointTarget.from_checkpoint()."
            )
        if self.format == "lora_adapter":
            raise NotImplementedError("LoRA adapters need Spindle or SGLang LoRA.")
        checkpoint = Checkpoint(
            checkpoint_type=CheckpointType.hf
            if self.format == "hf"
            else CheckpointType.megatron,
            name=self.path.rstrip("/").rsplit("/", 1)[-1],
            path=f"/checkpoints/{self.path}",
            timestamp=0.0,
            checkpoints_volume_name=self.volume_name,
            checkpoints_mount_path="/checkpoints",
        )
        endpoint = Endpoint.launch(
            self._model, checkpoint, environment=self.environment
        )
        try:
            endpoint.wait_until_ready()
            yield endpoint
        finally:
            endpoint.stop()

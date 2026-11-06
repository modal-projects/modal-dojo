# POC: option D — checkpoint as a first-class Modal Evals target

Not for merge. Built on PR 777 (`devin/1793377481-modal-evals`).

Callers pass a reference to weights on a Volume; Modal Evals owns serving
(convert, deploy, autoscale on trial concurrency, retry 429s, tear down).

```python
from modal_dojo.common.checkpoint_target import CheckpointTarget, ServingSpec
from modal_dojo.common.evaluations import run_evaluation

target = CheckpointTarget.from_checkpoint(
    run.latest_checkpoint(), dojo.models.Qwen3_4B,
    serving=ServingSpec(gpu="H100", min_containers=2),  # optional
)
job = run_evaluation("terminal-bench-2-1", target)
target.to_request()  # the proposed proto payload Modal Evals would accept
```

- `modal_dojo/common/checkpoint_target.py`: `CheckpointTarget`, `ServingSpec`,
  the proposed proto in the module docstring, and `serve()`, a client-side shim
  until Modal Evals implements it.
- `modal_dojo/common/evaluations.py`: `run_evaluation` accepts a
  `CheckpointTarget`; PR 777's `evaluate_checkpoint` no longer launches or stops
  Endpoints itself.

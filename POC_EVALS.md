# POC: option C — explicit `TrainingRun.evaluate()`

Not for merge. Built on PR 777 (`devin/1793377481-modal-evals`), which already
has `run_evaluation` / `publish_dataset_evaluation`.

The caller (user, notebook, or agent) decides when to evaluate. Dojo only
serves the checkpoint and records the result.

```python
run = dojo.TrainingRun.from_id("tr-...")
run.evaluate("terminal-bench-2-1", checkpoint="base")        # baseline
run.evaluate("terminal-bench-2-1")                           # latest checkpoint
run.evaluate("er-abc123", checkpoint="iter_0000040", trials_per_task=3)
run.evaluations()  # [{iteration, evaluation, job_id, status, score, url}, ...]
```

Change: `modal_dojo/common/run.py` (`TrainingRun.evaluate`, `_resolve_checkpoint`,
`_source_model`).

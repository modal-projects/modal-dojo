# POC: option B — Evals-owned registration

Not for merge. Built on PR 777 (`devin/1793377481-modal-evals`).

Dojo only declares what to evaluate. At training start it writes `eval.json`
next to the run's checkpoints and registers the checkpoints Volume. A
scheduler (which Modal Evals would own; stand-in here) finds new committed
checkpoints and evaluates the due ones. Any trainer that writes the same
manifest gets the same treatment.

```python
config = dojo.TrainConfig(
    dataset=gsm8k,
    model=dojo.models.Qwen3_4B,
    recipe=Qwen3_4B_Recipe(save_interval=20),
    eval_registration=EvalRegistration(
        evaluations=["aime-2026", "ifeval"],
        every_n_checkpoints=2,
        params={"temperature": 0.6, "max_tokens": 16384},
    ),
)
```

`<checkpoint_dir>/eval.json`:

```json
{
  "version": 1,
  "training_run_id": "tr-...",
  "base_model": "Qwen/Qwen3-4B",
  "evaluations": ["aime-2026", "ifeval"],
  "every_n_checkpoints": 2,
  "baseline": true,
  "params": {"temperature": 0.6, "max_tokens": 16384}
}
```

- `modal_dojo/common/eval_registration.py`: `EvalRegistration`, the
  `EvalManifest` contract, `write_manifest`, `register_volume`, `due_checkpoints`.
- `modal_dojo/evals_scheduler_poc.py`: the scheduler app (`tick` every 2 min,
  one container; `evaluate` per due checkpoint and evaluation).
- `train.py` / `launcher_helpers.py`: put the manifest on the run record at
  launch; write it and register the Volume when training starts.

# POC: option E — push `checkpoint_saved` events

Not for merge. Built on PR 777 (`devin/1793377481-modal-evals`).

The trainer's existing `checkpoint_save` phase report (slime and miles) also
puts an event on the named `modal.Queue` `modal-evals-checkpoint-events`,
partitioned by training run. PR 777's watcher blocks on that queue instead of
sleeping (fallback poll goes from 60 s to 10 min). Modal Evals could subscribe
to the same queue (or a webhook with the same payload).

```json
{"type": "checkpoint_saved", "version": 1, "training_run_id": "tr-...",
 "app_name": "...", "rollout_id": 39, "event_ts": 1793377481.0}
```

The hook fires before the checkpoint is committed, so the subscriber treats an
event as a wake-up and still confirms the checkpoint on the Volume.

- `modal_dojo/common/eval_events.py`: event contract, `maybe_emit_checkpoint_saved`,
  `wait_for_checkpoint_event`.
- `frameworks/{slime,miles}/phase_reporting.py`: emit from `report_step_event`.
- `common/evaluations.py`: `watch_checkpoints` wakes on events.

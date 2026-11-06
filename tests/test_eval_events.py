from modal_dojo.common import eval_events


def test_only_checkpoint_saves_become_events():
    assert eval_events.checkpoint_saved_event({"phase": "rollout"}) is None
    assert eval_events.checkpoint_saved_event({"phase": "checkpoint_save"}) is None
    event = eval_events.checkpoint_saved_event(
        {
            "phase": "checkpoint_save",
            "training_run_id": "run-1",
            "app_name": "app",
            "rollout_id": 39,
            "event_ts": 1.0,
        }
    )
    assert event == {
        "type": "checkpoint_saved",
        "version": 1,
        "training_run_id": "run-1",
        "app_name": "app",
        "rollout_id": 39,
        "event_ts": 1.0,
    }


def test_emit_never_raises(monkeypatch):
    class BrokenQueue:
        def put(self, *args, **kwargs):
            raise RuntimeError("queue down")

    monkeypatch.setattr(eval_events, "_queue", BrokenQueue)
    eval_events.maybe_emit_checkpoint_saved(
        {"phase": "checkpoint_save", "training_run_id": "run-1"}
    )

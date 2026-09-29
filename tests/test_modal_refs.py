from __future__ import annotations

import pickle

import cloudpickle
import modal
import pytest

from modal_dojo.common.modal_refs import (
    ModalCaptureError,
    _reduce_modal_sandbox,
    register_modal_cloudpickle_reducers,
)


def _modal_repr(value) -> str:
    original = next(
        attr_value
        for attr_name, attr_value in value.__dict__.items()
        if attr_name.startswith("_sync_original")
    )
    return original._rep


def test_modal_function_handle_fails_without_training_gym_reducer() -> None:
    helper = modal.Function.from_name("reward-helper", "score")

    with pytest.raises(AttributeError, match="_load_remote"):
        pickle.dumps(helper)


def test_captured_modal_function_from_name_survives_cloudpickle() -> None:
    register_modal_cloudpickle_reducers()
    helper = modal.Function.from_name(
        "reward-helper",
        "score",
        environment_name="main",
    )

    def custom_rm(prompt: str, response: str):
        return helper, prompt, response

    restored = cloudpickle.loads(cloudpickle.dumps(custom_rm))
    restored_helper, prompt, response = restored("p", "r")

    assert prompt == "p"
    assert response == "r"
    assert _modal_repr(restored_helper) == (
        "modal.Function.from_name('reward-helper', 'score', environment_name='main')"
    )


def test_inline_modal_function_survives_cloudpickle() -> None:
    register_modal_cloudpickle_reducers()
    app = modal.App("inline-probe")

    @app.function(serialized=True)
    def inline_fn(x: int = 1):
        return x + 1

    def custom_rm():
        return inline_fn

    restored = cloudpickle.loads(cloudpickle.dumps(custom_rm))
    restored_fn = restored()

    assert isinstance(restored_fn, modal.Function)
    assert "inline_fn" in _modal_repr(restored_fn)


def test_captured_modal_volume_and_secret_from_name_survive_cloudpickle() -> None:
    register_modal_cloudpickle_reducers()
    volume = modal.Volume.from_name("training-data", create_if_missing=True)
    secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])

    def custom_rm():
        return volume, secret

    restored = cloudpickle.loads(cloudpickle.dumps(custom_rm))
    restored_volume, restored_secret = restored()

    assert _modal_repr(restored_volume) == "modal.Volume.from_name('training-data')"
    assert _modal_repr(restored_secret) == "modal.Secret.from_name('wandb-secret')"


def test_captured_modal_name_based_handles_survive_cloudpickle() -> None:
    register_modal_cloudpickle_reducers()
    cls = modal.Cls.from_name("trainer-app", "Evaluator", environment_name="main")
    store = modal.Dict.from_name("scores", create_if_missing=True)
    queue = modal.Queue.from_name("jobs", environment_name="main")
    nfs = modal.NetworkFileSystem.from_name("shared-fs", create_if_missing=True)
    proxy = modal.Proxy.from_name("egress-proxy", environment_name="main")

    def custom_rm():
        return cls, store, queue, nfs, proxy

    restored = cloudpickle.loads(cloudpickle.dumps(custom_rm))
    restored_cls, restored_store, restored_queue, restored_nfs, restored_proxy = (
        restored()
    )

    assert _modal_repr(restored_cls) == (
        "Cls.from_name('trainer-app', 'Evaluator', environment_name='main')"
    )
    assert _modal_repr(restored_store) == "modal.Dict.from_name('scores')"
    assert _modal_repr(restored_queue) == (
        "modal.Queue.from_name('jobs', environment_name='main')"
    )
    assert _modal_repr(restored_nfs) == "NetworkFileSystem()"
    assert _modal_repr(restored_proxy) == (
        "modal.Proxy.from_name('egress-proxy', environment_name='main')"
    )


def test_captured_modal_id_based_handles_survive_cloudpickle() -> None:
    register_modal_cloudpickle_reducers()
    volume = modal.Volume.from_id("vo-123")
    store = modal.Dict.from_id("di-123")
    queue = modal.Queue.from_id("qu-123")
    call = modal.FunctionCall.from_id("fc-123")
    image = modal.Image.from_id("im-123")
    snapshot = modal.SandboxSnapshot.from_id("ss-123")

    def custom_rm():
        return volume, store, queue, call, image, snapshot

    restored = cloudpickle.loads(cloudpickle.dumps(custom_rm))
    (
        restored_volume,
        restored_store,
        restored_queue,
        restored_call,
        restored_image,
        restored_snapshot,
    ) = restored()

    assert _modal_repr(restored_volume) == "Volume.from_id('vo-123')"
    assert _modal_repr(restored_store) == "Dict.from_id('di-123')"
    assert _modal_repr(restored_queue) == "Queue.from_id('qu-123')"
    assert _modal_repr(restored_call) == "FunctionCall.from_id('fc-123')"
    assert _modal_repr(restored_image) == "Image.from_id('im-123')"
    assert _modal_repr(restored_snapshot) == "SandboxSnapshot()"


def test_modal_sandbox_reducer_uses_object_id() -> None:
    class FakeSandbox:
        _object_id = "sb-123"

    restore, args = _reduce_modal_sandbox(FakeSandbox())

    assert restore.__name__ == "_restore_modal_id_handle"
    assert args == ("Sandbox", "sb-123")


def test_unnameable_modal_handle_gets_clear_error(monkeypatch) -> None:
    register_modal_cloudpickle_reducers()
    helper = modal.Function.from_name("reward-helper", "score")
    original = next(
        attr_value
        for attr_name, attr_value in helper.__dict__.items()
        if attr_name.startswith("_sync_original")
    )
    monkeypatch.setattr(original, "_load", None)

    with pytest.raises(ModalCaptureError, match="Function.from_name"):
        cloudpickle.dumps(helper)

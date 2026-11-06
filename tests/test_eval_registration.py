import json

from modal_dojo.common.eval_registration import (
    MANIFEST_METADATA_KEY,
    MANIFEST_NAME,
    EvalManifest,
    due_checkpoints,
    write_manifest,
)


def _manifest(**overrides) -> EvalManifest:
    return EvalManifest(
        evaluations=["aime-2026"],
        training_run_id="run-1",
        base_model="Qwen/Qwen3-4B",
        **overrides,
    )


def test_due_checkpoints_includes_baseline_and_every_nth():
    names = [f"iter_{i:07d}" for i in (10, 20, 30, 40)]
    manifest = _manifest(every_n_checkpoints=2)
    assert due_checkpoints(manifest, names, started=set()) == [
        "base",
        "iter_0000020",
        "iter_0000040",
    ]
    assert due_checkpoints(manifest, names, started={"base", "iter_0000020"}) == [
        "iter_0000040"
    ]


def test_write_manifest_keeps_the_first_one(tmp_path):
    first = {MANIFEST_METADATA_KEY: {"training_run_id": "run-1"}}
    write_manifest(first, str(tmp_path))
    write_manifest(
        {MANIFEST_METADATA_KEY: {"training_run_id": "resumed"}}, str(tmp_path)
    )
    assert json.loads((tmp_path / MANIFEST_NAME).read_text()) == {
        "training_run_id": "run-1"
    }

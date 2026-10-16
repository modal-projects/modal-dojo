"""Launch one H200 proof after all calls in an existing conversion tree finish.

Run from the autoconfig checkout. Uses current local Dojo sources and does not
stop apps, cancel calls, or modify the existing conversion lock.
"""

import json
import time
from pathlib import Path

import modal

from autoconfig.app import image

APP_NAME = "muse-spark-h200-64k-validation-v2"
SHARED_NAME = "miles-muse_spark_open_64k_b300_recipe"
SOURCE_COMMIT = "1338f57"
app = modal.App(APP_NAME, image=image)


def pending_calls(graph, expected_call_id):
    """Call graphs include ancestors; inspect every node, including retried children."""
    nodes = []

    def visit(items):
        for node in items:
            nodes.append(node)
            visit(node.children)

    visit(graph)
    if not any(n.function_call_id == expected_call_id for n in nodes):
        raise RuntimeError(
            f"Expected conversion {expected_call_id} absent from call graph"
        )
    from modal.call_graph import InputStatus

    return [n for n in nodes if n.status == InputStatus.PENDING]


def checkpoint_wait_reason(holder, expected_owner, now, ttl):
    if not holder:
        return None
    owner = str(holder.get("run_id") or "")
    if owner != expected_owner:
        return f"checkpoint owner changed to {owner!r}; preserving its lock"
    claimed_at = holder.get("claimed_at")
    if not isinstance(claimed_at, (int, float)) or now - claimed_at < ttl:
        return f"checkpoint lock still fresh for {owner}"
    return None


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("_training-gym-modal-creds")],
    cpu=0.5,
    memory=2048,
    timeout=86400,
    retries=0,
)
def validate(request, after_call, expected_conversion_call, expected_owner):
    from modal_dojo.frameworks.miles.launcher import (
        _CONVERT_LOCK_TTL_S,
        _convert_lock_dict,
        _convert_lock_key,
    )

    from autoconfig.api import RunRequest, launch_run, plan_run

    cfg = plan_run(RunRequest.model_validate(request))
    assert cfg.recipe.gpu_type == "H200" and cfg.recipe.gpu_allocation.total_gpus == 64
    assert cfg.recipe.seq_length == 65536 and cfg.recipe.num_rollout == 1
    assert cfg.recipe.name == SHARED_NAME
    assert cfg.recipe.save is None and cfg.recipe.save_interval is None
    call = modal.FunctionCall.from_id(after_call)
    locks = _convert_lock_dict()
    key = _convert_lock_key(SHARED_NAME + "-checkpoints", cfg.recipe.ref_load)
    quiet_checks = 0
    last_message = None
    deadline = time.monotonic() + 20 * 3600
    print(
        "H200_VALIDATION_QUEUED",
        json.dumps(
            {
                "source_commit": SOURCE_COMMIT,
                "expected_conversion_call": expected_conversion_call,
                "steps": 1,
                "gpus": 64,
                "checkpoint_volume": SHARED_NAME + "-checkpoints",
            }
        ),
        flush=True,
    )
    while time.monotonic() < deadline:
        pending = pending_calls(call.get_call_graph(), expected_conversion_call)
        if pending:
            reason = "pending dependency calls: " + ", ".join(
                n.function_name + "/" + n.function_call_id for n in pending
            )
        else:
            reason = checkpoint_wait_reason(
                locks.get(key, None), expected_owner, time.time(), _CONVERT_LOCK_TTL_S
            )
        if reason:
            quiet_checks = 0
            if reason != last_message:
                print("WAITING_WITHOUT_STOPPING", reason, flush=True)
                last_message = reason
        else:
            quiet_checks += 1
            if quiet_checks >= 2:
                break
            print("DEPENDENCIES_QUIET_VERIFYING_AGAIN", flush=True)
        time.sleep(30)
    else:
        raise TimeoutError(
            "Existing preparation is still active; no H200 GPU run submitted"
        )
    print("H200_PREPARATION_STARTING", flush=True)
    result = launch_run(request)
    print("H200_TRAINING_SUBMITTED", json.dumps(result), flush=True)
    return result


@app.local_entrypoint()
def main():
    from autoconfig.api import RunRequest, encode_recipe, plan_run
    from examples.muse_spark_64k import CONFIG, build_run

    record = Path("/private/tmp/muse-h200-validation-v2-launch.json")
    assert not record.exists(), (
        "Submission record already exists; refusing duplicate validation"
    )
    run = build_run(gpu="H200", steps=1)
    run.recipe.capture_trace = True
    run.recipe.trace_sample_limit = 4
    run.metadata.update(
        {
            "stage": "one-step-proof",
            "attempt": 5,
            "controller": APP_NAME,
            "dojo_commit": SOURCE_COMMIT,
            "checkpoint_preparation": "wait-for-all-dependency-calls",
            "checkpoint_volume": SHARED_NAME + "-checkpoints",
            "image_validation_app": "ap-JG12XVG13IDsMWKqaCvpIr",
        }
    )
    req = RunRequest(
        group=CONFIG.to_json(),
        recipe=encode_recipe(run.recipe),
        recipe_class=type(run.recipe).__name__,
        metadata=run.metadata,
        dry_run=False,
    )
    assert plan_run(req).recipe.name == SHARED_NAME
    call = validate.spawn(
        req.model_dump(mode="json"),
        "fc-01M52S3RY43XAS382TD485MF4C",
        "fc-01M52YD50B1DKETS2G22AZ5XY4",
        "asymptotic-similarity-89353a872d87",
    )
    info = {
        "app_id": app.app_id,
        "call_id": call.object_id,
        "source_commit": SOURCE_COMMIT,
        "gpu": "H200",
        "gpus": 64,
        "steps": 1,
        "context_length": 65536,
        "waits_for_conversion": "fc-01M52YD50B1DKETS2G22AZ5XY4",
    }
    record.write_text(json.dumps(info, indent=2) + "\n")
    print("H200_VALIDATION_SUBMITTED", json.dumps(info), flush=True)
    result = call.get()
    Path("/private/tmp/muse-h200-validation-v2-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)

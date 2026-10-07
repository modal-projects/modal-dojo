"""Allow DeepSeek to skip Miles' automatic engine weight-hash audit.

In Miles ea751aac8, engine registration returns 202 before the router finishes
discovering metadata. The following synchronous SGLang checksum blocks engine
responses long enough for every registration job to time out. These hashes are
audit records, not trainer-versus-engine equality checks. Keep explicit weight
checks available and let this recipe opt out of the automatic audit at every
weight update, including startup.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/ray/placement_group.py")
MARKER = "PATCHED_DEEPSEEK_V41_CHECKSUM_AUDIT"
OLD = """async def _maybe_log_inference_engine_weight_checksums(
    args, *, inference_controller: BaseWorkerHandle, rollout_id: int | None, trainer_model_id: str | None
) -> None:
"""
NEW = (
    OLD
    + f"""    # {MARKER}: allow expensive audit hashes to be disabled per recipe.
    import os

    if os.environ.get("MILES_SKIP_ENGINE_WEIGHT_CHECKSUM") == "1":
        return

"""
)


def main() -> None:
    source = TARGET.read_text()
    if MARKER not in source:
        if source.count(OLD) != 1:
            raise ValueError("DeepSeek checksum audit patch did not match")
        patched = source.replace(OLD, NEW, 1)
        compile(patched, str(TARGET), "exec")
        TARGET.write_text(patched)
    print("Patched optional engine weight checksum audit")


if __name__ == "__main__":
    main()

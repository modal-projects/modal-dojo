"""Make expensive engine audit hashes optional for the DeepSeek recipe.

Upstream's default DeepSeek launcher does not save checkpoints or audit events,
so its post-sync checksum is skipped. Dojo saves checkpoints and thus implicitly
enables audit events. Allow this recipe to skip just the hash pass while keeping
checkpointing and other events. Preserve hashes when the event analyzer needs
them, including when upstream enables that analyzer for fault tolerance.
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
    + f"""    # {MARKER}: keep checkpointing independent of expensive audit hashes.
    import os

    if (
        os.environ.get("MILES_SKIP_ENGINE_WEIGHT_CHECKSUM") == "1"
        and not getattr(args, "enable_event_analyzer", False)
    ):
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

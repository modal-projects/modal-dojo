from pathlib import Path

TARGET = Path("/root/miles/miles/backends/training_utils/weight_update/updater.py")
OLD = """                materialize=protocol.is_sender,
            ):
                if protocol.is_sender:
                    if driver and checksums is not None:
                        record_lora_checksums(bucket, checksums)
                    protocol.send_bucket(bucket)
"""
NEW = """                materialize=protocol.is_sender or (driver and checksums is not None),
            ):
                if driver and checksums is not None:
                    record_lora_checksums(bucket, checksums)
                if protocol.is_sender:
                    protocol.send_bucket(bucket)
"""


def patch_source(source: str) -> str:
    if source.count(NEW) == 1:
        return source
    if source.count(OLD) != 1:
        raise ValueError("Miles LoRA checksum driver patch anchor changed")
    source = source.replace(OLD, NEW, 1)
    compile(source, str(TARGET), "exec")
    return source


if __name__ == "__main__":
    TARGET.write_text(patch_source(TARGET.read_text()))
    print(f"Patched {TARGET}: build LoRA checksums even when rank zero is not a sender")

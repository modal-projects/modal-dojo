"""Separate inference readiness from transfer time in the pinned Miles sync path.

Source: miles@41c5e38b94ea23677de93b01a4a77d55677a8f09.
"""

from pathlib import Path

TARGET = Path("/root/miles/miles/ray/train/group.py")
MARKER = "DOJO_WEIGHT_SYNC"
OPERATIONS = (
    (
        "prepare_engines",
        "        info = await self._inference_controller.start_update_weights()\n",
    ),
    (
        "transfer",
        """        weight_versions = await retry(
            lambda _: self._execute_first_alive("update_weights", info=info),
            max_attempts=_RETRY_MAX_ATTEMPTS,
        )
""",
    ),
    (
        "finish_engines",
        "        await self._inference_controller.end_update_weights(snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes)\n",
    ),
)


def apply(target: Path = TARGET) -> None:
    source = target.read_text()
    if MARKER in source:
        return
    for phase, old in OPERATIONS:
        if source.count(old) != 1:
            raise RuntimeError(f"{target}: weight-sync source changed at {phase}")
        new = (
            "        import time as _dojo_time\n"
            "        _dojo_started = _dojo_time.monotonic()\n"
            "        _dojo_ok = False\n"
            "        try:\n"
            + "".join("    " + line for line in old.splitlines(keepends=True))
            + "            _dojo_ok = True\n"
            "        finally:\n"
            f'            logger.info("{MARKER} phase={phase} ok=%s elapsed_s=%.3f", '
            "_dojo_ok, _dojo_time.monotonic() - _dojo_started)\n"
        )
        source = source.replace(old, new)
    compile(source, str(target), "exec")
    target.write_text(source)


if __name__ == "__main__":
    apply()

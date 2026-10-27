"""Cross-run lock for converting a checkpoint into a shared ``ref_load``."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any

from modal import Dict as ModalDict


_CONVERT_LOCK_DICT_NAME = "training-gym-convert-lock"
_CONVERT_LOCK_TTL_S = 900.0
_CONVERT_LOCK_REFRESH_S = 300.0
_CONVERT_TOKEN_TTL_S = 4 * _CONVERT_LOCK_TTL_S


def _convert_lock_dict() -> Any:
    return ModalDict.from_name(_CONVERT_LOCK_DICT_NAME, create_if_missing=True)


def _convert_lock_key(volume_name: str, save_path: str) -> str:
    return f"lock:{volume_name}:{save_path}"


def _sweep_convert_tokens(locks: Any, key: str) -> None:
    """Drop takeover tokens old enough that no live contender can act on one.

    A token has to outlive the takeover it settles, or a contender still holding the
    stale read it was written for could win a fresh one and pop the new owner's claim.
    Past that it is only litter, and nothing else reclaims it, so tokens are swept on
    release and on the next takeover for the same path.
    """
    prefix = f"{key}:takeover:"
    cutoff = time.time() - _CONVERT_TOKEN_TTL_S
    try:
        names = [str(name) for name in locks.keys() if str(name).startswith(prefix)]
        for name in names:
            token = locks.get(name)
            written_at = token.get("at") if isinstance(token, dict) else None
            if not isinstance(written_at, (int, float)) or written_at < cutoff:
                locks.pop(name, None)
    except Exception as exc:
        print(f"WARNING: could not sweep stale conversion tokens: {exc}")


def acquire_convert_lock(run_id: str, volume_name: str, save_path: str) -> str:
    """Claim the right to convert into ``save_path``; return the current holder.

    Two launches of the same recipe share one ``ref_load``, so without this the
    loser's cleanup deletes the winner's still-metadata-less conversion output.

    ``Dict.put(..., skip_if_exists=True)`` is the atomic compare-and-set that decides
    the winner; a read-then-write claim would let two runs both observe an unheld lock
    and both proceed. The claim expires after ``_CONVERT_LOCK_TTL_S``, sized against the
    heartbeat interval so a holder that cannot run its release path frees the path in a
    TTL rather than a conversion's worth of time.

    Taking a stale claim over cannot be a delete followed by a put: two runs that both
    read the same stale holder would delete each other's fresh claim and both come away
    believing they own the path. Instead each contender first claims a token naming the
    stale holder it saw, which only one can win, and only that winner replaces the
    claim. Those tokens are swept by age, not on use — see ``_sweep_convert_tokens``.
    """
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    claim = {"run_id": run_id, "claimed_at": time.time()}
    if locks.put(key, claim, skip_if_exists=True):
        return run_id

    owner, claimed_at = "", None
    holder = locks.get(key)
    if isinstance(holder, dict):
        owner = str(holder.get("run_id") or "")
        claimed_at = holder.get("claimed_at")
        if owner == run_id:
            return run_id
        if (
            owner
            and isinstance(claimed_at, (int, float))
            and time.time() - claimed_at < _CONVERT_LOCK_TTL_S
        ):
            return owner

    takeover_key = f"{key}:takeover:{owner}:{claimed_at}"
    token = {"run_id": run_id, "at": time.time()}
    if not locks.put(takeover_key, token, skip_if_exists=True):
        winner = locks.get(takeover_key)
        if isinstance(winner, dict) and winner.get("run_id"):
            return str(winner["run_id"])
        return owner or run_id

    _sweep_convert_tokens(locks, key)
    try:
        locks.pop(key)
    except KeyError:
        pass
    if locks.put(key, claim, skip_if_exists=True):
        return run_id
    holder = locks.get(key)
    if isinstance(holder, dict) and holder.get("run_id"):
        return str(holder["run_id"])
    return run_id


def refresh_convert_lock(run_id: str, volume_name: str, save_path: str) -> None:
    """Extend an existing claim of this run's; never create one.

    A refresh that wrote unconditionally would resurrect the claim when a tick
    straddles the release, leaving it held with nobody to give it back.
    """
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    holder = locks.get(key)
    if not isinstance(holder, dict) or str(holder.get("run_id") or "") != run_id:
        return
    locks[key] = {"run_id": run_id, "claimed_at": time.time()}


@contextlib.contextmanager
def convert_lock_heartbeat(run_id: str, volume_name: str, save_path: str):
    """Keep this run's claim fresh for as long as the body runs.

    Refreshing on elapsed time rather than on work done: a torch_dist conversion
    writes only about one file per rank, so any progress-based cadence never fires,
    and the claim would lapse mid-conversion and let a later launch delete the output
    being written.
    """
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(_CONVERT_LOCK_REFRESH_S):
            try:
                refresh_convert_lock(run_id, volume_name, save_path)
            except Exception as exc:
                print(f"WARNING: could not refresh conversion claim: {exc}")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5.0)


def release_convert_lock(run_id: str, volume_name: str, save_path: str) -> None:
    """Drop the conversion claim, but only if this run still holds it."""
    locks = _convert_lock_dict()
    key = _convert_lock_key(volume_name, save_path)
    holder = locks.get(key)
    if isinstance(holder, dict) and str(holder.get("run_id") or "") != run_id:
        return
    try:
        locks.pop(key)
    except KeyError:
        pass
    _sweep_convert_tokens(locks, key)

"""Enum-backed helpers for metadata stored in the shared Modal Volume."""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from functools import partial
from typing import Any, Literal, TypeVar, cast, overload

from modal_dojo._api_reference import exclude_from_api_reference

T = TypeVar("T")

METADATA_VOLUME_NAME = "modal-dojo-metadata"
_READ_CONCURRENCY = 16


@exclude_from_api_reference
class MetadataStore(Enum):
    """Named prefixes for JSON records on the shared metadata volume."""

    TRAINING_RUNS = "training-runs"
    TRAINING_RUNS_SUMMARY = "training-runs-summary"
    FRAMEWORK_STATUS_TOKENS = "framework-status-tokens"
    TRAIN_RESULTS = "train-results"
    TRAIN_RESULTS_SUMMARY = "train-results-summary"
    TRAINING_ROLLOUTS = "training-rollouts"
    TRAINING_ROLLOUTS_SUMMARY = "training-rollouts-summary"
    # Per-step, per-group advantage distributions. slime only logs the mean
    # advantage per step; this store keeps the full per-sample distribution so
    # the dashboard can render per-group spread. Written one shard file per
    # data-parallel rank (keyed ``{run}__{rollout:08d}__dp{dp:03d}``) so
    # concurrent DP-rank posts never race on a shared file.
    ADVANTAGE_DISTRIBUTIONS = "advantage-distributions"
    TRAINING_RUN_UPDATES = "training-run-updates"
    EVAL_RESULTS = "eval-results"
    EVALS = "evals"
    EVAL_SUMMARIES = "eval-summaries"
    EVAL_CONFIGS = "eval-configs"
    SUBSTEP_TIMING = "substep-timing"
    # Scalar metric histories mirrored from the framework's W&B-style logger.
    # One sub-directory per run holding step-range chunk files, see
    # ``common/metric_series.py``.
    METRIC_SERIES = "metric-series"


SUMMARY_KEY = "summary"
SUMMARY_ITEMS_KEY = "items"


# Summary stores keep one JSON file per item, named after the canonical item's
# key, so concurrent writers can never race on a shared file. The legacy
# single ``summary`` file written by older code is still read for items that
# were never rewritten by a new writer; healed reads compare canonical keys
# against summary ids in both directions and rebuild/drop until the summary
# converges to the canonical set. ``item_id_key`` names the payload field that
# carries the canonical file's key. ``project`` is a ``module:attr`` path to a
# canonical-payload -> summary-item reducer for stores whose canonical shape
# differs from the summary row (resolved lazily to avoid import cycles).
class _SummaryCompaction:
    __slots__ = ("item_store", "item_id_key", "sort_key", "reverse", "project")

    def __init__(self, item_store, item_id_key, sort_key, reverse, project=None):
        self.item_store = item_store
        self.item_id_key = item_id_key
        self.sort_key = sort_key
        self.reverse = reverse
        self.project = project


_SUMMARY_COMPACTION: dict[MetadataStore, _SummaryCompaction] = {
    MetadataStore.TRAINING_RUNS_SUMMARY: _SummaryCompaction(
        item_store=MetadataStore.TRAINING_RUNS,
        item_id_key="training_run_id",
        sort_key=lambda item: (
            int(item.get("created_at", 0) or 0),
            str(item.get("training_run_id", "")),
        ),
        reverse=True,
    ),
    MetadataStore.TRAIN_RESULTS_SUMMARY: _SummaryCompaction(
        item_store=MetadataStore.TRAIN_RESULTS,
        item_id_key="training_run_id",
        sort_key=lambda item: str(item.get("training_run_id", "")),
        reverse=True,
    ),
    MetadataStore.EVAL_SUMMARIES: _SummaryCompaction(
        item_store=MetadataStore.EVAL_RESULTS,
        item_id_key="eval_id",
        sort_key=lambda item: str(item.get("created_at", "")),
        reverse=True,
        project="modal_dojo.common.eval:eval_summary_item",
    ),
}


def _summary_projection(
    cfg: _SummaryCompaction,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if cfg.project is None:
        return lambda payload: payload
    import importlib

    module_name, _, attr = cfg.project.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def _metadata_volume():
    import modal

    return modal.Volume.from_name(METADATA_VOLUME_NAME, create_if_missing=True)


def _safe_reload(vol, *, is_async: bool = False):
    if is_async:

        async def _run() -> None:
            try:
                await vol.reload.aio()
            except RuntimeError:
                pass

        return _run()
    try:
        vol.reload()
    except RuntimeError:
        pass


def _store_path(store: MetadataStore | str) -> str:
    if isinstance(store, MetadataStore):
        return store.value
    return store


async def bounded_gather_with_retries(
    readers: Iterable[Callable[[], Awaitable[T]]],
) -> list[T | BaseException]:
    from modal.exception import Error

    semaphore = asyncio.Semaphore(_READ_CONCURRENCY)

    async def _read(reader: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            for attempt in range(3):
                try:
                    return await reader()
                except Error as exc:
                    if "rate limit" not in str(exc).lower() or attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    return await asyncio.gather(
        *(_read(reader) for reader in readers),
        return_exceptions=True,
    )


def vol_remove(store: MetadataStore | str, key: str) -> bool:
    """Delete a single item from a store. Returns True if removed."""
    from modal.exception import InvalidError, NotFoundError

    vol = _metadata_volume()
    path = f"{_store_path(store)}/{key}.json"
    try:
        vol.remove_file(path)
        return True
    except (FileNotFoundError, NotFoundError):
        return False
    except InvalidError as exc:
        if "No such file or directory" in str(exc):
            return False
        raise


@overload
def vol_put(
    store: MetadataStore | str,
    key: str,
    value: dict[str, Any],
    *,
    is_async: Literal[True],
) -> Awaitable[None]: ...


@overload
def vol_put(
    store: MetadataStore | str,
    key: str,
    value: dict[str, Any],
    *,
    is_async: Literal[False] = False,
) -> None: ...


def vol_put(
    store: MetadataStore | str,
    key: str,
    value: dict[str, Any],
    *,
    is_async: bool = False,
) -> None | Awaitable[None]:
    # force=True overwrites in place at commit, so the file is never absent
    # mid-write. A remove-then-upload sequence leaves a window where readers
    # see no file and treat the store as empty — which silently collapses
    # read-modify-write summaries down to a single item.
    vol = _metadata_volume()
    data = json.dumps(value).encode()
    path = f"{_store_path(store)}/{key}.json"
    if is_async:

        async def _run() -> None:
            async with vol.batch_upload(force=True) as batch:
                batch.put_file(io.BytesIO(data), path)

        return _run()
    with vol.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), path)


@overload
def vol_put_many(
    store: MetadataStore | str,
    values: dict[str, dict[str, Any]],
    *,
    is_async: Literal[True],
) -> Awaitable[None]: ...


@overload
def vol_put_many(
    store: MetadataStore | str,
    values: dict[str, dict[str, Any]],
    *,
    is_async: Literal[False] = False,
) -> None: ...


def vol_put_many(
    store: MetadataStore | str,
    values: dict[str, dict[str, Any]],
    *,
    is_async: bool = False,
) -> None | Awaitable[None]:
    """Write several keys from one store in a single volume commit."""
    vol = _metadata_volume()
    data = {
        f"{_store_path(store)}/{key}.json": json.dumps(value).encode()
        for key, value in values.items()
    }
    if is_async:

        async def _run() -> None:
            async with vol.batch_upload(force=True) as batch:
                for path, payload in data.items():
                    batch.put_file(io.BytesIO(payload), path)

        return _run()
    with vol.batch_upload(force=True) as batch:
        for path, payload in data.items():
            batch.put_file(io.BytesIO(payload), path)


@overload
def vol_get(
    store: MetadataStore | str,
    key: str,
    *,
    is_async: Literal[True],
) -> Awaitable[dict[str, Any]]: ...


@overload
def vol_get(
    store: MetadataStore | str,
    key: str,
    *,
    is_async: Literal[False] = False,
) -> dict[str, Any]: ...


def vol_get(
    store: MetadataStore | str, key: str, *, is_async: bool = False
) -> dict[str, Any] | Awaitable[dict[str, Any]]:
    vol = _metadata_volume()
    path = f"{_store_path(store)}/{key}.json"
    if is_async:

        async def _run() -> dict[str, Any]:
            try:
                chunks = [chunk async for chunk in vol.read_file.aio(path)]
                return json.loads(b"".join(chunks))
            except FileNotFoundError:
                await _safe_reload(vol, is_async=True)
            try:
                chunks = [chunk async for chunk in vol.read_file.aio(path)]
                return json.loads(b"".join(chunks))
            except FileNotFoundError:
                raise KeyError(key) from None

        return _run()
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except FileNotFoundError:
        _safe_reload(vol)
    try:
        return json.loads(b"".join(vol.read_file(path)))
    except FileNotFoundError:
        raise KeyError(key) from None


@overload
def vol_list(
    store: MetadataStore | str,
    *,
    is_async: Literal[True],
) -> Awaitable[list[dict[str, Any]]]: ...


@overload
def vol_list(
    store: MetadataStore | str,
    *,
    is_async: Literal[False] = False,
) -> list[dict[str, Any]]: ...


def vol_list(
    store: MetadataStore | str,
    *,
    is_async: bool = False,
) -> list[dict[str, Any]] | Awaitable[list[dict[str, Any]]]:
    result = _vol_list_core(store, is_async=is_async)
    if is_async:

        async def _run() -> list[dict[str, Any]]:
            records, failure = await cast(
                Awaitable[tuple[list[dict[str, Any]], BaseException | None]], result
            )
            if failure is not None:
                raise failure
            return records

        return _run()
    records, failure = cast(tuple[list[dict[str, Any]], BaseException | None], result)
    if failure is not None:
        raise failure
    return records


_LIST_ATTEMPTS = 3


def _is_rate_limit(exc: BaseException) -> bool:
    return "rate limit" in str(exc).lower()


def _metadata_entry(entry: Any) -> dict[str, Any]:
    return {
        "path": entry.path,
        "mtime": entry.mtime,
        "size": entry.size,
    }


def _list_metadata_entries(
    store: MetadataStore | str,
    *,
    is_async: bool = False,
) -> (
    tuple[list[dict[str, Any]], BaseException | None]
    | Awaitable[tuple[list[dict[str, Any]], BaseException | None]]
):
    from modal.exception import Error, NotFoundError

    vol = _metadata_volume()
    if is_async:

        async def _run() -> tuple[list[dict[str, Any]], BaseException | None]:
            await _safe_reload(vol, is_async=True)
            for attempt in range(_LIST_ATTEMPTS):
                try:
                    entries = [
                        _metadata_entry(entry)
                        async for entry in vol.iterdir.aio(_store_path(store))
                        if entry.path.endswith(".json")
                    ]
                    return entries, None
                except (FileNotFoundError, NotFoundError):
                    return [], None
                except Error as exc:
                    if not _is_rate_limit(exc) or attempt == _LIST_ATTEMPTS - 1:
                        return [], exc
                    await asyncio.sleep(2**attempt)
            raise AssertionError("unreachable")

        return _run()

    _safe_reload(vol)
    for attempt in range(_LIST_ATTEMPTS):
        try:
            return [
                _metadata_entry(entry)
                for entry in vol.iterdir(_store_path(store))
                if entry.path.endswith(".json")
            ], None
        except (FileNotFoundError, NotFoundError):
            return [], None
        except Error as exc:
            if not _is_rate_limit(exc) or attempt == _LIST_ATTEMPTS - 1:
                return [], exc
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


@overload
def _read_metadata_records(
    entries: list[dict[str, Any]],
    *,
    is_async: Literal[True],
) -> Awaitable[tuple[list[dict[str, Any]], BaseException | None]]: ...


@overload
def _read_metadata_records(
    entries: list[dict[str, Any]],
    *,
    is_async: Literal[False] = False,
) -> tuple[list[dict[str, Any]], BaseException | None]: ...


def _read_metadata_records(
    entries: list[dict[str, Any]],
    *,
    is_async: bool = False,
) -> (
    tuple[list[dict[str, Any]], BaseException | None]
    | Awaitable[tuple[list[dict[str, Any]], BaseException | None]]
):
    from modal.exception import Error, NotFoundError

    vol = _metadata_volume()
    if is_async:

        async def _read(path: str) -> dict[str, Any] | None:
            try:
                chunks = [chunk async for chunk in vol.read_file.aio(path)]
                return json.loads(b"".join(chunks))
            except (FileNotFoundError, NotFoundError):
                return None
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None

        async def _run() -> tuple[list[dict[str, Any]], BaseException | None]:
            results = await bounded_gather_with_retries(
                [lambda entry=entry: _read(entry["path"]) for entry in entries]
            )
            records: list[dict[str, Any]] = []
            failure: BaseException | None = None
            for entry, result in zip(entries, results, strict=True):
                if result is None:
                    continue
                if isinstance(result, BaseException):
                    failure = failure or result
                    continue
                if not isinstance(result, dict):
                    continue
                records.append(result)
            return records, failure

        return _run()

    def _read_sync(path: str) -> Any:
        for attempt in range(_LIST_ATTEMPTS):
            try:
                return json.loads(b"".join(vol.read_file(path)))
            except (
                FileNotFoundError,
                NotFoundError,
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                return None
            except Error as exc:
                if not _is_rate_limit(exc) or attempt == _LIST_ATTEMPTS - 1:
                    return exc
                time.sleep(2**attempt)

    records: list[dict[str, Any]] = []
    failure: BaseException | None = None
    with ThreadPoolExecutor(max_workers=_READ_CONCURRENCY) as pool:
        results = list(pool.map(_read_sync, [entry["path"] for entry in entries]))
    for result in results:
        if result is None:
            continue
        if isinstance(result, BaseException):
            failure = failure or result
            continue
        if not isinstance(result, dict):
            continue
        records.append(result)
    return records, failure


@overload
def _vol_list_core(
    store: MetadataStore | str,
    *,
    is_async: Literal[True],
) -> Awaitable[tuple[list[dict[str, Any]], BaseException | None]]: ...


@overload
def _vol_list_core(
    store: MetadataStore | str,
    *,
    is_async: Literal[False] = False,
) -> tuple[list[dict[str, Any]], BaseException | None]: ...


def _vol_list_core(
    store: MetadataStore | str,
    *,
    is_async: bool = False,
) -> (
    tuple[list[dict[str, Any]], BaseException | None]
    | Awaitable[tuple[list[dict[str, Any]], BaseException | None]]
):
    entries = _list_metadata_entries(store, is_async=is_async)
    if is_async:

        async def _run() -> tuple[list[dict[str, Any]], BaseException | None]:
            entries_result, failure = await cast(
                Awaitable[tuple[list[dict[str, Any]], BaseException | None]], entries
            )
            if failure is not None:
                return [], failure
            return await _read_metadata_records(entries_result, is_async=True)

        return _run()
    entries_result, failure = cast(
        tuple[list[dict[str, Any]], BaseException | None], entries
    )
    if failure is not None:
        return [], failure
    return _read_metadata_records(entries_result)


@overload
def vol_list_metadata_with_failures(
    store: MetadataStore | str,
    *,
    is_async: Literal[True],
) -> Awaitable[tuple[list[dict[str, Any]], bool]]: ...


@overload
def vol_list_metadata_with_failures(
    store: MetadataStore | str,
    *,
    is_async: Literal[False] = False,
) -> tuple[list[dict[str, Any]], bool]: ...


def vol_list_metadata_with_failures(
    store: MetadataStore | str,
    *,
    is_async: bool = False,
) -> tuple[list[dict[str, Any]], bool] | Awaitable[tuple[list[dict[str, Any]], bool]]:
    entries = _list_metadata_entries(store, is_async=is_async)
    if is_async:

        async def _run() -> tuple[list[dict[str, Any]], bool]:
            entries_result, failure = await cast(
                Awaitable[tuple[list[dict[str, Any]], BaseException | None]], entries
            )
            return entries_result, failure is not None

        return _run()
    entries_result, failure = cast(
        tuple[list[dict[str, Any]], BaseException | None], entries
    )
    return entries_result, failure is not None


def vol_list_keys(store: MetadataStore | str, prefix: str = "") -> list[str]:
    """List matching JSON keys without downloading their payloads."""
    entries, failure = _list_metadata_entries(store)
    if failure is not None:
        raise failure
    return [
        key
        for entry in entries
        if (key := entry["path"].rsplit("/", 1)[-1][:-5]).startswith(prefix)
    ]


def vol_list_prefix(store: MetadataStore | str, prefix: str) -> list[dict[str, Any]]:
    """Read only the items whose key (file basename) starts with ``prefix``.

    Lists directory entries (cheap, no payload reads) and fetches only the
    matching files. Used to gather the per-DP-rank shards of one
    ``(run, rollout)`` without reading the whole store.
    """
    from modal.exception import NotFoundError

    vol = _metadata_volume()
    _safe_reload(vol)
    results: list[dict[str, Any]] = []
    try:
        for entry in vol.iterdir(_store_path(store)):
            if not entry.path.endswith(".json"):
                continue
            name = entry.path.rsplit("/", 1)[-1][: -len(".json")]
            if not name.startswith(prefix):
                continue
            results.append(json.loads(b"".join(vol.read_file(entry.path))))
    except (FileNotFoundError, NotFoundError):
        return results
    return results


def vol_remove_keys_with_prefix(store: MetadataStore | str, prefix: str) -> int:
    """Delete every item whose key (file basename) starts with ``prefix``.

    Reads directory entries only (unlike vol_list_prefix). Returns the number of items removed.
    """
    from modal.exception import NotFoundError

    vol = _metadata_volume()
    _safe_reload(vol)
    try:
        entries = list(vol.iterdir(_store_path(store)))
    except (FileNotFoundError, NotFoundError):
        return 0
    removed = 0
    for entry in entries:
        name = entry.path.rsplit("/", 1)[-1]
        if not name.endswith(".json") or not name.startswith(prefix):
            continue
        if vol_remove(store, name[: -len(".json")]):
            removed += 1
    return removed


def vol_count_items(store: MetadataStore | str) -> int:
    """Count canonical ``.json`` files in a store without reading them.

    A single directory listing, used to cheaply detect a collapsed summary
    (summary item count < canonical file count) before paying for a full
    rebuild via ``vol_list``.
    """
    from modal.exception import NotFoundError

    vol = _metadata_volume()
    _safe_reload(vol)
    try:
        return sum(
            1 for e in vol.iterdir(_store_path(store)) if e.path.endswith(".json")
        )
    except (FileNotFoundError, NotFoundError):
        return 0


def compact_summary_store(summary_store: MetadataStore) -> list[dict[str, Any]]:
    """Rebuild a registered summary from its canonical per-item files."""
    cfg = _SUMMARY_COMPACTION[summary_store]
    return vol_compact_summary_items(
        summary_store,
        cfg.item_store,
        item_id_key=cfg.item_id_key,
        project=_summary_projection(cfg),
        sort_key=cfg.sort_key,
        reverse=cfg.reverse,
    )


def vol_get_summary_items_healed(
    summary_store: MetadataStore, *, prefix: str = ""
) -> list[dict[str, Any]]:
    """Read a summary, converging it to the canonical key set first.

    Per-item summary files make lost updates impossible, but items written
    before this layout (or never summarized) only exist in the canonical
    store, and deleted canonical items leave stale summary rows behind. One
    directory listing per side finds both directions of drift; missing items
    are rebuilt (and back-filled as per-item files) and stale ones pruned, so
    every reader after the first sees the converged summary.
    """
    items = vol_get_summary_items(summary_store, prefix=prefix) or []
    cfg = _SUMMARY_COMPACTION.get(summary_store)
    if cfg is None:
        return items
    canonical_keys = set(vol_list_keys(cfg.item_store, prefix))
    by_id = {
        str(item[cfg.item_id_key]): item
        for item in items
        if item.get(cfg.item_id_key) is not None
    }
    missing = sorted(canonical_keys - by_id.keys())
    stale = sorted(by_id.keys() - canonical_keys)
    if not missing and not stale:
        return items
    project = _summary_projection(cfg)
    for key in missing:
        try:
            payload = vol_get(cfg.item_store, key)
        except KeyError:
            continue
        item = project(payload)
        if not isinstance(item, dict):
            continue
        item.setdefault(cfg.item_id_key, key)
        by_id[key] = item
        try:
            vol_put(summary_store, key, item)
        except Exception:
            pass
    for item_id in stale:
        by_id.pop(item_id, None)
        vol_remove(summary_store, item_id)
    items = list(by_id.values())
    items.sort(key=cfg.sort_key, reverse=cfg.reverse)
    if not prefix:
        try:
            vol_put_summary_items(summary_store, items, item_id_key=cfg.item_id_key)
        except Exception:
            pass
    return items


def summary_items_from_payload(
    payload: Any,
    payload_key: str = SUMMARY_ITEMS_KEY,
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get(payload_key, [])
    else:
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _summary_entry_names(
    entries: list[dict[str, Any]], key: str, prefix: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    item_entries: list[dict[str, Any]] = []
    legacy_entry: dict[str, Any] | None = None
    for entry in entries:
        name = entry["path"].rsplit("/", 1)[-1][: -len(".json")]
        if name == key:
            legacy_entry = entry
        elif not prefix or name.startswith(prefix):
            item_entries.append(entry)
    return item_entries, legacy_entry


def _merge_summary_records(
    records: list[dict[str, Any]],
    legacy_items: list[dict[str, Any]],
    id_key: str | None,
    prefix: str,
    cfg: _SummaryCompaction | None,
) -> list[dict[str, Any]]:
    items = [record for record in records if isinstance(record, dict)]
    have = (
        {str(item[id_key]) for item in items if item.get(id_key) is not None}
        if id_key
        else set()
    )
    for item in legacy_items:
        if not isinstance(item, dict):
            continue
        item_id = item.get(id_key) if id_key else None
        if item_id is None:
            continue
        item_id = str(item_id)
        if prefix and not item_id.startswith(prefix):
            continue
        if item_id in have:
            continue
        items.append(item)
        have.add(item_id)
    if cfg is not None:
        items.sort(key=cfg.sort_key, reverse=cfg.reverse)
    return items


@overload
def vol_get_summary_items(
    store: MetadataStore | str,
    *,
    item_id_key: str | None = None,
    prefix: str = "",
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    is_async: Literal[True],
) -> Awaitable[list[dict[str, Any]] | None]: ...


@overload
def vol_get_summary_items(
    store: MetadataStore | str,
    *,
    item_id_key: str | None = None,
    prefix: str = "",
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    is_async: Literal[False] = False,
) -> list[dict[str, Any]] | None: ...


def vol_get_summary_items(
    store: MetadataStore | str,
    *,
    item_id_key: str | None = None,
    prefix: str = "",
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    is_async: bool = False,
) -> list[dict[str, Any]] | None | Awaitable[list[dict[str, Any]] | None]:
    """Read a summary: per-item files, then the legacy shared file fills gaps.

    Per-item files are authoritative (each is written by its item's owner);
    the legacy ``summary`` file only contributes items that newer writers
    never materialized. ``prefix`` restricts both sides to keys/ids sharing a
    prefix. Returns ``None`` when the store holds nothing at all.
    """
    cfg = _SUMMARY_COMPACTION.get(store) if isinstance(store, MetadataStore) else None
    id_key = item_id_key or (cfg.item_id_key if cfg is not None else None)
    listed = _list_metadata_entries(store, is_async=is_async)

    async def _run(
        entries_result: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        item_entries, legacy_entry = _summary_entry_names(entries_result, key, prefix)
        records, _ = await _read_metadata_records(item_entries, is_async=True)
        legacy_items: list[dict[str, Any]] = []
        if legacy_entry is not None:
            legacy_records, legacy_failure = await _read_metadata_records(
                [legacy_entry], is_async=True
            )
            if legacy_failure is not None:
                print(f"WARNING: unreadable summary {store}/{key}: {legacy_failure}")
            if legacy_records:
                legacy_items = summary_items_from_payload(
                    legacy_records[0], payload_key
                )
        if not item_entries and legacy_entry is None:
            return None
        return _merge_summary_records(records, legacy_items, id_key, prefix, cfg)

    if is_async:

        async def _await() -> list[dict[str, Any]] | None:
            entries_result, failure = await cast(
                Awaitable[tuple[list[dict[str, Any]], BaseException | None]], listed
            )
            if failure is not None:
                raise failure
            return await _run(entries_result)

        return _await()
    entries_result, failure = cast(
        tuple[list[dict[str, Any]], BaseException | None], listed
    )
    if failure is not None:
        raise failure
    item_entries, legacy_entry = _summary_entry_names(entries_result, key, prefix)
    records, _ = _read_metadata_records(item_entries)
    legacy_items = []
    if legacy_entry is not None:
        legacy_records, legacy_failure = _read_metadata_records([legacy_entry])
        if legacy_failure is not None:
            print(f"WARNING: unreadable summary {store}/{key}: {legacy_failure}")
        if legacy_records:
            legacy_items = summary_items_from_payload(legacy_records[0], payload_key)
    if not item_entries and legacy_entry is None:
        return None
    return _merge_summary_records(records, legacy_items, id_key, prefix, cfg)


def vol_put_summary_items(
    store: MetadataStore | str,
    items: list[dict[str, Any]],
    *,
    item_id_key: str | None = None,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    is_async: bool = False,
    prune: bool = False,
) -> None | Awaitable[None]:
    """Replace a summary's contents: per-item files plus the legacy list file.

    The per-item files are written first so the shared file never advertises
    items the per-item layout lacks; afterwards, files for items absent from
    ``items`` are pruned so a replace-shaped write stays exact.
    """
    cfg = _SUMMARY_COMPACTION.get(store) if isinstance(store, MetadataStore) else None
    id_key = item_id_key or (cfg.item_id_key if cfg is not None else None)
    per_item: dict[str, dict[str, Any]] = {}
    if id_key:
        for item in items:
            if isinstance(item, dict) and item.get(id_key) is not None:
                per_item[str(item[id_key])] = item

    def _prune_stale() -> None:
        try:
            names = set(vol_list_keys(store))
        except Exception:
            return
        for name in names - set(per_item) - {key}:
            vol_remove(store, name)

    if is_async:

        async def _run() -> None:
            if per_item:
                await vol_put_many(store, per_item, is_async=True)
            await vol_put(store, key, {payload_key: items}, is_async=True)
            if prune:
                await asyncio.to_thread(_prune_stale)

        return _run()
    if per_item:
        vol_put_many(store, per_item)
    vol_put(store, key, {payload_key: items})
    if prune:
        _prune_stale()


def vol_compact_summary_items(
    summary_store: MetadataStore | str,
    item_store: MetadataStore | str,
    *,
    item_id_key: str,
    project: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    key: str = SUMMARY_KEY,
    payload_key: str = SUMMARY_ITEMS_KEY,
    sort_key: Callable[[dict[str, Any]], Any] | None = None,
    reverse: bool = False,
) -> list[dict[str, Any]]:
    """Rebuild a denormalized summary from canonical per-item metadata files.

    The current summary contents seed the rebuild (covering items whose
    canonical read fails); canonical payloads then overlay them. ``project``
    reduces a canonical payload to its summary row for stores whose canonical
    shape is richer than the summary's; it defaults to identity.
    """
    project = project or (lambda payload: payload)
    summary_items = vol_get_summary_items(
        summary_store, item_id_key=item_id_key, key=key, payload_key=payload_key
    )
    canonical_items, failure = _vol_list_core(item_store)
    if failure is not None and summary_items is None:
        raise failure

    items_by_id = {
        item[item_id_key]: item
        for item in summary_items or []
        if item.get(item_id_key) is not None
    }
    for canonical in canonical_items:
        item = project(canonical)
        if not isinstance(item, dict):
            continue
        item_id = item.get(item_id_key)
        if item_id is None:
            continue
        items_by_id[item_id] = {**items_by_id.get(item_id, {}), **item}

    items = list(items_by_id.values())
    if sort_key is not None:
        items.sort(key=sort_key, reverse=reverse)
    vol_put_summary_items(
        summary_store,
        items,
        item_id_key=item_id_key,
        key=key,
        payload_key=payload_key,
        prune=True,
    )
    if failure is not None:
        raise failure
    return items


def vol_put_with_summary(
    item_store: MetadataStore | str,
    key: str,
    payload: dict[str, Any],
    *,
    summary_store: MetadataStore | str,
    summary_item: dict[str, Any] | None = None,
    is_async: bool = False,
) -> None | Awaitable[None]:
    """Persist a canonical item file, then its per-item summary file.

    Canonical first (source of truth), then the summary row under the same
    key — a single-owner single-file write, so concurrent writers can never
    collapse a shared summary file. ``summary_item`` defaults to ``payload``
    for stores whose summary rows mirror the canonical shape.
    """
    put = partial(vol_put, item_store, key, payload)
    put_summary = partial(
        vol_put,
        summary_store,
        key,
        payload if summary_item is None else summary_item,
    )
    if is_async:

        async def _run() -> None:
            await put(is_async=True)
            await put_summary(is_async=True)

        return _run()
    put()
    put_summary()


__all__ = [
    "METADATA_VOLUME_NAME",
    "MetadataStore",
    "SUMMARY_ITEMS_KEY",
    "SUMMARY_KEY",
    "summary_items_from_payload",
    "bounded_gather_with_retries",
    "vol_get",
    "vol_list",
    "vol_list_keys",
    "vol_list_prefix",
    "vol_count_items",
    "compact_summary_store",
    "vol_get_summary_items_healed",
    "vol_put",
    "vol_put_many",
    "vol_remove",
    "vol_get_summary_items",
    "vol_put_summary_items",
    "vol_put_with_summary",
    "vol_compact_summary_items",
]

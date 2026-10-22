import time

import pytest

from modal_dojo.common import convert_lock

_MISSING = object()


class FakeDict:
    """In-memory stand-in for the ``modal.Dict`` calls the lock makes."""

    def __init__(self) -> None:
        self.data: dict[str, object] = {}

    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.data:
            return False
        self.data[key] = value
        return True

    def get(self, key, default=None):
        return self.data.get(key, default)

    def pop(self, key, default=_MISSING):
        if key in self.data:
            return self.data.pop(key)
        if default is _MISSING:
            raise KeyError(key)
        return default

    def keys(self):
        return list(self.data)

    def __setitem__(self, key, value):
        self.data[key] = value


@pytest.fixture
def locks(monkeypatch):
    fake = FakeDict()
    monkeypatch.setattr(convert_lock, "_convert_lock_dict", lambda: fake)
    return fake


VOL, PATH = "slime-qwen-checkpoints", "/checkpoints/model_torch_dist"


def test_first_claim_wins_and_others_see_the_holder(locks):
    assert convert_lock.acquire_convert_lock("run-a", VOL, PATH) == "run-a"
    assert convert_lock.acquire_convert_lock("run-b", VOL, PATH) == "run-a"
    assert convert_lock.acquire_convert_lock("run-a", VOL, PATH) == "run-a"


def test_paths_and_volumes_lock_independently(locks):
    assert convert_lock.acquire_convert_lock("run-a", VOL, PATH) == "run-a"
    assert convert_lock.acquire_convert_lock("run-b", VOL, PATH + "_tp2") == "run-b"
    assert convert_lock.acquire_convert_lock("run-c", "other-vol", PATH) == "run-c"


def test_only_the_holder_releases(locks):
    convert_lock.acquire_convert_lock("run-a", VOL, PATH)
    convert_lock.release_convert_lock("run-b", VOL, PATH)
    assert convert_lock.acquire_convert_lock("run-b", VOL, PATH) == "run-a"

    convert_lock.release_convert_lock("run-a", VOL, PATH)
    assert convert_lock.acquire_convert_lock("run-b", VOL, PATH) == "run-b"


def test_expired_claim_is_taken_over(locks):
    key = convert_lock._convert_lock_key(VOL, PATH)
    stale = time.time() - convert_lock._CONVERT_LOCK_TTL_S - 1
    locks.data[key] = {"run_id": "crashed", "claimed_at": stale}

    assert convert_lock.acquire_convert_lock("run-b", VOL, PATH) == "run-b"
    assert convert_lock.acquire_convert_lock("run-c", VOL, PATH) == "run-b"


def test_refresh_never_creates_or_steals_a_claim(locks):
    key = convert_lock._convert_lock_key(VOL, PATH)
    convert_lock.refresh_convert_lock("run-a", VOL, PATH)
    assert key not in locks.data

    convert_lock.acquire_convert_lock("run-a", VOL, PATH)
    convert_lock.refresh_convert_lock("run-b", VOL, PATH)
    assert locks.data[key]["run_id"] == "run-a"

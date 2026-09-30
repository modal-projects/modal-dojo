"""An image a reward function stashes on ``sample.metadata["image"]`` (e.g. a
rendering of the model's output) flows through rollout extraction: compacted by
``RolloutImageStore`` and re-emitted as ``metadata["image"]``/``image_ref`` so
the dashboard renders it per sample."""

import base64

from modal_dojo.common.sample_extraction import (
    RolloutImageStore,
    _sample_to_dict,
)

# A 1x1 red PNG. Real bytes so the test exercises the same path whether PIL is
# installed (decode + thumbnail) or not (small data-URI passthrough).
_RED_PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/q842"
    "iQAAAABJRU5ErkJggg=="
)
_BLUE_PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYPj/HwADAgH/p2/z"
    "ngAAAABJRU5ErkJggg=="
)


def _data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _sample(png: bytes, **extra) -> dict:
    return {
        "prompt": "draw an eye",
        "response": "code",
        "reward": 1.0,
        "metadata": {"image": _data_uri(png), **extra},
    }


def test_reward_output_image_annotated():
    store = RolloutImageStore(16)
    meta = _sample_to_dict(_sample(_RED_PX), image_store=store)["metadata"]
    assert meta["_metadata_type"] == "image"
    assert meta["image"].startswith("data:image/")
    assert meta["image_ref"]


def test_distinct_output_images_get_distinct_refs():
    store = RolloutImageStore(16)
    a = _sample_to_dict(_sample(_RED_PX), image_store=store)["metadata"]
    b = _sample_to_dict(_sample(_BLUE_PX), image_store=store)["metadata"]
    assert a["image_ref"] != b["image_ref"]
    assert "image" in a and "image" in b


def test_duplicate_output_image_carries_bytes_once():
    store = RolloutImageStore(16)
    first = _sample_to_dict(_sample(_RED_PX), image_store=store)["metadata"]
    second = _sample_to_dict(_sample(_RED_PX), image_store=store)["metadata"]
    assert first["image_ref"] == second["image_ref"]
    assert "image" in first and "image" not in second


def test_other_metadata_still_passes_through():
    store = RolloutImageStore(16)
    meta = _sample_to_dict(_sample(_RED_PX, judge="pairwise"), image_store=store)[
        "metadata"
    ]
    assert meta["judge"] == "pairwise"


def test_no_image_metadata_untouched():
    store = RolloutImageStore(16)
    sample = {"prompt": "p", "response": "r", "reward": 0.0, "metadata": {"k": "v"}}
    meta = _sample_to_dict(sample, image_store=store)["metadata"]
    assert "image" not in meta and "image_ref" not in meta
    assert meta.get("_metadata_type") != "image"

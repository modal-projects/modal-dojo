from __future__ import annotations

from modal_dojo.common.dataset import OnlineRollout


def test_online_rollout_rows() -> None:
    dataset = OnlineRollout(n_rows=4)
    assert list(dataset.rows()) == [
        {"prompt": [{"role": "user", "content": ""}], "label": "0"},
        {"prompt": [{"role": "user", "content": ""}], "label": "1"},
        {"prompt": [{"role": "user", "content": ""}], "label": "2"},
        {"prompt": [{"role": "user", "content": ""}], "label": "3"},
    ]
    assert dataset.cache_key() == "online-rollout-4-conversation"

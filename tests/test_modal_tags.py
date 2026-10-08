from modal_dojo.common import modal_tag_value


def test_modal_tag_value_uses_raw_model_name() -> None:
    assert modal_tag_value("Qwen/Qwen3.6-35B-A3B") == "qwen3.6-35b-a3b"


def test_modal_tag_value_replaces_non_name_chars() -> None:
    assert modal_tag_value("project/foo bar:baz") == "foo-bar-baz"

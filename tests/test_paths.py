from pathlib import Path

import pytest

from ssl_probe.paths import dataset_utt_id_overrides, utterance_relative_path


def test_bare_utterance_id_is_placed_under_split() -> None:
    assert utterance_relative_path("1234", "picnic") == Path("picnic/1234")


def test_split_qualified_utterance_id_is_not_duplicated() -> None:
    assert utterance_relative_path("picnic/1234", "picnic") == Path("picnic/1234")


def test_nested_utterance_id_is_preserved_without_split() -> None:
    assert utterance_relative_path("picnic/1234", None) == Path("picnic/1234")


@pytest.mark.parametrize("utt_id", ["/absolute/1234", "../outside"])
def test_unsafe_utterance_id_is_rejected(utt_id: str) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        utterance_relative_path(utt_id, "picnic")


def test_named_dataset_keeps_its_configured_utterance_template() -> None:
    assert dataset_utt_id_overrides("clac", None) == {}


def test_generic_dataset_defaults_to_filename_stem() -> None:
    assert dataset_utt_id_overrides(None, None) == {"utt_id_template": "{path.stem}"}


def test_explicit_utterance_template_overrides_dataset_config() -> None:
    assert dataset_utt_id_overrides("clac", "{path.name}") == {"utt_id_template": "{path.name}"}

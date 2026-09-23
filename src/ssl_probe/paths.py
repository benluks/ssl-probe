from pathlib import Path


def utterance_relative_path(utt_id: str, split: str | None) -> Path:
    """Return one dataset-relative path without duplicating a qualified split."""
    path = Path(utt_id)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Utterance ID must be a safe relative path, got {utt_id!r}.")

    if split is None or (path.parts and path.parts[0] == split):
        return path

    return Path(split) / path


def dataset_utt_id_overrides(
    dataset_name: str | None,
    utt_id_template: str | None,
) -> dict[str, str]:
    """Preserve named-dataset IDs while giving generic datasets stable IDs."""
    if utt_id_template is not None:
        return {"utt_id_template": utt_id_template}
    if dataset_name is None:
        return {"utt_id_template": "{path.stem}"}
    return {}

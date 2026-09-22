from pathlib import Path


def utterance_relative_path(utt_id: str, split: str | None) -> Path:
    """Return one dataset-relative path without duplicating a qualified split."""
    path = Path(utt_id)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Utterance ID must be a safe relative path, got {utt_id!r}.")

    if split is None or (path.parts and path.parts[0] == split):
        return path

    return Path(split) / path

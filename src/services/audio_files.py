"""Shared validation helpers for uploaded audio content."""

from pathlib import PurePosixPath

DEFAULT_AUDIO_EXTENSIONS = (".flac", ".m4a", ".mp3", ".ogg", ".wav")


def validate_audio_bytes(
    filename: str,
    size: int,
    *,
    allowed_extensions: tuple[str, ...] = DEFAULT_AUDIO_EXTENSIONS,
    max_size_bytes: int = 20 * 1024 * 1024,
) -> str | None:
    """Return a user-safe validation error, or ``None`` when the upload is valid.

    `filename`/`size` describe an already-buffered upload (content-based,
    not a server-local path) — there is no directory-containment check
    because uploaded bytes never live at a caller-supplied path.
    """
    suffix = PurePosixPath(filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        return "The audio format is not supported."
    if size <= 0:
        return "The audio file is empty."
    if size > max_size_bytes:
        return "The audio file exceeds the configured size limit."
    return None

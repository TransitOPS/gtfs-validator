"""GTFS input abstraction: ZIP, directory, and URL loaders."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import IO, Protocol

from gtfs_validator.notices import Notice, Severity

# Maximum in-memory download size (2 GB).
_MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024

# Patterns filtered from ZIP archives.
_SKIP_PREFIXES = ("__MACOSX/",)
_SKIP_NAMES = {".DS_Store"}


class GtfsInput(Protocol):
    """Abstract contract for reading a GTFS feed."""

    def filenames(self) -> set[str]: ...
    def open_file(self, filename: str) -> IO[bytes]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Directory input
# ---------------------------------------------------------------------------


class DirectoryInput:
    """Reads a GTFS feed from an unarchived directory (flat layout)."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._files: dict[str, Path] = {}
        for entry in directory.iterdir():
            if entry.is_file():
                self._files[entry.name] = entry

    def filenames(self) -> set[str]:
        return set(self._files)

    def open_file(self, filename: str) -> IO[bytes]:
        return open(self._files[filename], "rb")

    def close(self) -> None:
        pass  # no resources to release

    def __enter__(self) -> DirectoryInput:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ZIP input
# ---------------------------------------------------------------------------


def _safe_entry_name(name: str) -> str | None:
    """Return the sanitized basename, or ``None`` if the entry should be skipped."""
    if name.endswith("/"):
        return None
    for prefix in _SKIP_PREFIXES:
        if name.startswith(prefix):
            return None
    basename = name.rsplit("/", 1)[-1]
    if basename in _SKIP_NAMES:
        return None
    # Zip-slip prevention.
    if ".." in name or name.startswith("/"):
        return None
    return basename


class ZipInput:
    """Reads a GTFS feed from a ZIP archive (file or in-memory bytes)."""

    def __init__(
        self,
        source: Path | io.BytesIO,
        *,
        notices: list[Notice] | None = None,
    ) -> None:
        self._zf = zipfile.ZipFile(source)
        self._entries: dict[str, str] = {}  # basename -> zip entry name
        subdirectory_warned = False

        for info in self._zf.infolist():
            basename = _safe_entry_name(info.filename)
            if basename is None:
                continue
            # Detect files in subdirectories.
            if "/" in info.filename and notices is not None and not subdirectory_warned:
                notices.append(
                    Notice(
                        code="subdirectory_transit_feed",
                        severity=Severity.INFO,
                        fields={"filename": info.filename},
                    )
                )
                subdirectory_warned = True
            if basename not in self._entries:
                self._entries[basename] = info.filename

    def filenames(self) -> set[str]:
        return set(self._entries)

    def open_file(self, filename: str) -> IO[bytes]:
        return self._zf.open(self._entries[filename])

    def close(self) -> None:
        self._zf.close()

    def __enter__(self) -> ZipInput:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def open_input(
    gtfs_source: str,
    storage_directory: Path | None = None,
) -> tuple[GtfsInput, list[Notice]]:
    """Open a GTFS feed from a local path or URL.

    Returns the input handle and any notices emitted during opening.
    """
    notices: list[Notice] = []

    if gtfs_source.startswith(("http://", "https://")):
        return _open_url(gtfs_source, storage_directory, notices), notices

    path = Path(gtfs_source)
    if path.is_dir():
        return DirectoryInput(path), notices
    # Assume ZIP file.
    return ZipInput(path, notices=notices), notices


def _open_url(
    url: str,
    storage_directory: Path | None,
    notices: list[Notice],
) -> GtfsInput:
    import httpx

    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        response = client.get(url)
        response.raise_for_status()
        data = response.content
        if len(data) > _MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Downloaded feed exceeds {_MAX_DOWNLOAD_BYTES} bytes"
            )

    if storage_directory is not None:
        storage_directory.mkdir(parents=True, exist_ok=True)
        # Derive filename from URL or use default.
        filename = url.rsplit("/", 1)[-1] or "feed.zip"
        dest = storage_directory / filename
        dest.write_bytes(data)
        return ZipInput(dest, notices=notices)

    return ZipInput(io.BytesIO(data), notices=notices)

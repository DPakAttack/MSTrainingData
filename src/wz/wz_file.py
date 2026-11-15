"""PKG1 `.wz` package reader (port of the PKG1 paths in `Wz_File.cs` / `Wz_Structure.cs`).

Only the PKG1 format is supported. That covers both classic single-file clients and the newer split
layout where each folder holds a stub `X.wz`, an `X.ini` with `LastWzIndex`, and data in `X_NNN.wz`.
"""

from __future__ import annotations

import logging
import mmap
import re
from dataclasses import dataclass, field
from pathlib import Path

from wz.binary_reader import WzReader
from wz.crypto import WzKey, pkg1_hash_version, pkg1_offset, pkg1_version_candidates
from wz.wz_image import KeyHint, WzImage

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[\w.\-# ()+&'!@$%^,\[\]{}~=]+$", re.UNICODE)
_MAX_VERSION_CHECKS = 8


@dataclass
class WzEntry:
    """An image or directory inside a package's directory tree."""

    name: str
    """Entry name."""

    is_dir: bool
    """Whether this entry is a directory."""

    size: int
    """Size of the image data in bytes (images only)."""

    checksum: int
    """Stored checksum (images only)."""

    offset_pos: int
    """File position of the hashed offset field."""

    hashed_offset: int
    """Encrypted data offset."""

    children: list[WzEntry] = field(default_factory=list)
    """Child entries (directories only)."""

    offset: int = 0
    """Decrypted data offset (images only, filled in after version detection)."""


class VersionHint:
    """Shared "last version hash that worked" across files in one structure."""

    def __init__(self) -> None:
        self.hashes: list[int] = []


class WzFile:
    """A single PKG1 `.wz` file opened with a read-only memory map.

    Parameters
    ----------
    path
        Path to the `.wz` file.
    key_hint
        Shared key preference.
    version_hint
        Shared version-hash preference.
    """

    def __init__(self, path: Path, key_hint: KeyHint, version_hint: VersionHint) -> None:
        self.path = path
        self._key_hint = key_hint
        self._version_hint = version_hint
        self._fh = path.open("rb")
        size = path.stat().st_size
        if size < 16:
            self._fh.close()
            raise ValueError(f"{path} is too small to be a wz file")
        self.data = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = size
        self._read_header()
        self.key: WzKey | None = None
        self.hash_version: int | None = None
        self.entries: list[WzEntry] = self._read_tree()

    def close(self) -> None:
        """Close the memory map and file handle.

        If views into the map are still alive, the map is left for garbage collection.
        """
        try:
            self.data.close()
        except BufferError:
            pass
        self._fh.close()

    def _read_header(self) -> None:
        r = WzReader(self.data)
        signature = r.read(4)
        if signature != b"PKG1":
            raise ValueError(f"{self.path}: unsupported signature {signature!r} (only PKG1 is supported)")
        self.data_size = r.i64()
        self.header_size = r.i32()
        encver_missing = False
        self.encrypted_version = -1
        if self.data_size >= 2:
            r.pos = self.header_size
            encver = r.u16()
            if encver > 0xFF:
                encver_missing = True
            elif encver == 0x80 and self.data_size >= 5:
                r.pos = self.header_size
                count = r.cint()
                if count > 0 and (count & 0xFF) == 0 and count <= 0xFFFF:
                    encver_missing = True
            if not encver_missing:
                self.encrypted_version = encver
        else:
            encver_missing = True
        self.encver_missing = encver_missing
        self.dir_start = self.header_size + (0 if encver_missing else 2)

    # Directory tree -----------------------------------------------------------------------

    def _read_tree(self) -> list[WzEntry]:
        errors: list[str] = []
        for key in self._key_hint.candidates():
            try:
                entries = self._read_dir(WzReader(self.data, self.dir_start), key)
            except (ValueError, IndexError, EOFError, UnicodeDecodeError) as exc:
                errors.append(f"{key.key_type.value}: {exc}")
                continue
            if all(_NAME_RE.match(e.name) for e in _walk(entries)):
                self.key = key
                self._key_hint.key = key
                self._resolve_offsets(entries)
                return entries
            errors.append(f"{key.key_type.value}: names did not decode cleanly")
        raise ValueError(f"{self.path}: could not read directory ({'; '.join(errors)})")

    def _read_dir(self, r: WzReader, key: WzKey) -> list[WzEntry]:
        entries: list[WzEntry] = []
        count = r.cint()
        if count < 0 or count > 1_000_000:
            raise ValueError(f"implausible entry count {count}")
        for _ in range(count):
            node_type = r.u8()
            if node_type == 0x01:
                # Unknown/placeholder entry: int32 + int16 of padding, plus size/checksum/offset.
                r.skip(10)
                continue
            if node_type == 0x02:
                add = 2 if self.encver_missing else -1
                name = r.string_at(self.dir_start + r.i32() + add, key)
                is_dir = False
            elif node_type in (0x03, 0x04):
                name = r.string(key)
                is_dir = node_type == 0x03
            else:
                raise ValueError(f"unknown dir entry type {node_type}")
            size = r.cint()
            checksum = r.cint()
            pos = r.pos
            hashed = r.u32()
            entries.append(WzEntry(name, is_dir, size, checksum, pos, hashed))
        for entry in entries:
            if entry.is_dir:
                entry.children = self._read_dir(r, key)
        return entries

    def _resolve_offsets(self, entries: list[WzEntry]) -> None:
        images = [e for e in _walk(entries) if not e.is_dir]
        if not images:
            return
        for hash_version in self._version_candidates():
            if self._check_version(images, hash_version):
                self.hash_version = hash_version
                if hash_version in self._version_hint.hashes:
                    self._version_hint.hashes.remove(hash_version)
                self._version_hint.hashes.insert(0, hash_version)
                for img in images:
                    img.offset = pkg1_offset(img.offset_pos, img.hashed_offset, self.header_size, hash_version)
                return
        raise ValueError(f"{self.path}: could not detect the wz version")

    def _version_candidates(self) -> list[int]:
        if self.encver_missing:
            candidates = [pkg1_hash_version(777)]
        else:
            candidates = [h for _, h in pkg1_version_candidates(self.encrypted_version)]
        preferred = [h for h in self._version_hint.hashes if h in candidates]
        return preferred + [h for h in candidates if h not in preferred]

    def _check_version(self, images: list[WzEntry], hash_version: int) -> bool:
        step = max(1, len(images) // _MAX_VERSION_CHECKS)
        for img in images[::step][:_MAX_VERSION_CHECKS]:
            off = pkg1_offset(img.offset_pos, img.hashed_offset, self.header_size, hash_version)
            if off < self.dir_start or off + img.size > self.size:
                return False
            if self.data[off] not in (0x73, 0x1B, 0x23, 0x52):  # binary tags, or text "#Property"/"Root"
                return False
        return True

    def make_image(self, entry: WzEntry) -> WzImage:
        """Create a lazily-parsed `WzImage` for an image entry."""
        data = self.data
        offset = entry.offset

        def loader() -> tuple[memoryview, int]:
            return memoryview(data), offset

        return WzImage(entry.name, loader, entry.size, self._key_hint)


def _walk(entries: list[WzEntry]) -> list[WzEntry]:
    out: list[WzEntry] = []
    stack = list(entries)
    while stack:
        e = stack.pop()
        out.append(e)
        stack.extend(e.children)
    return out


def read_last_wz_index(ini_path: Path) -> int | None:
    """Parse `LastWzIndex|N` from a split package `.ini` file."""
    if not ini_path.is_file():
        return None
    for line in ini_path.read_text(encoding="latin-1").splitlines():
        key, _, value = line.partition("|")
        if key.strip() == "LastWzIndex":
            try:
                return int(value.strip())
            except ValueError:
                return None
    return None


def package_files(folder: Path) -> list[Path]:
    """List the `.wz` files that make up the split package in `folder`.

    The entry file is `<folder>/<name>.wz`; data files are `<name>_000.wz` .. `<name>_NNN.wz` where
    `NNN` comes from `<name>.ini` (or from probing when the `.ini` is missing).
    """
    name = folder.name
    files = []
    entry = folder / f"{name}.wz"
    if entry.is_file():
        files.append(entry)
    last = read_last_wz_index(folder / f"{name}.ini")
    if last is None:
        i = 0
        while (folder / f"{name}_{i:03d}.wz").is_file():
            i += 1
        last = i - 1
    for i in range(last + 1):
        extra = folder / f"{name}_{i:03d}.wz"
        if extra.is_file():
            files.append(extra)
    return files

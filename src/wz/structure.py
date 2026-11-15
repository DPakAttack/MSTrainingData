"""Mount a client `Data` directory as one lazily-loaded `WzNode` tree (port of `Wz_Structure.cs`).

Layout handled:

- Split packages: a folder `X/` holding `X.wz` (+ `X.ini`, `X_000.wz` ...). Nested package folders
  (e.g. `Map/Map/Map0/`, `Mob/_Canvas/`) are mounted at the matching path.
- Classic single files: `X.wz` directly inside the root.
- Encrypted packs: `Packs/*.ms` (or `*.ms` in the root). Their entries carry full paths such as
  `Mob/0100100.img` and are overlaid onto the tree.

Nothing is read until a node's children are first accessed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from types import TracebackType

from wz.ms_file import MsEntry, MsFileBase, open_ms_file
from wz.node import WzNode
from wz.wz_file import VersionHint, WzEntry, WzFile, package_files
from wz.wz_image import KeyHint

logger = logging.getLogger(__name__)


def _is_package_folder(folder: Path) -> bool:
    name = folder.name
    return (folder / f"{name}.wz").is_file() or (folder / f"{name}_000.wz").is_file()


class WzStructure:
    """The virtual tree over every package below `root_dir`.

    Parameters
    ----------
    root_dir
        Client data directory, e.g. `.../Data`.

    Examples
    --------
    >>> with WzStructure(Path("Data")) as wz:
    ...     wz.find("String/Mob.img/100100/name").value
    'Snail'
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"WZ root directory not found: {self.root_dir}")
        self.root = WzNode("")
        self._key_hint = KeyHint()
        self._version_hint = VersionHint()
        self._open_files: list[WzFile] = []
        self._ms_entries: dict[str, list[tuple[MsFileBase, MsEntry]]] = defaultdict(list)
        self._open_ms: list[MsFileBase] = []
        self._scan()

    # Lifecycle ----------------------------------------------------------------------------

    def close(self) -> None:
        """Close every open file and drop the mounted tree."""
        self.root = WzNode("")
        for f in self._open_files:
            f.close()
        for ms in self._open_ms:
            ms.close()
        self._open_files.clear()
        self._open_ms.clear()

    def __enter__(self) -> WzStructure:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    # Queries ------------------------------------------------------------------------------

    def find(self, path: str) -> WzNode | None:
        """Find a node by slash-separated path from the root, e.g. `Mob/0100100.img/stand/0`."""
        return self.root.find(path)

    # Mounting -----------------------------------------------------------------------------

    def _scan(self) -> None:
        ms_paths = sorted((self.root_dir / "Packs").glob("*.ms")) + sorted(self.root_dir.glob("*.ms"))
        for path in ms_paths:
            try:
                ms = open_ms_file(path, self._key_hint)
            except ValueError as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
            self._open_ms.append(ms)
            for entry in ms.entries:
                if entry.name.lower().endswith(".img") and "/" in entry.name:
                    top = entry.name.split("/", 1)[0]
                    self._ms_entries[top].append((ms, entry))

        mounted: set[str] = set()
        for folder in sorted(p for p in self.root_dir.iterdir() if p.is_dir()):
            if _is_package_folder(folder):
                self._mount_lazy(self.root.add(folder.name), folder=folder, top=True)
                mounted.add(folder.name)
        for wz_path in sorted(self.root_dir.glob("*.wz")):
            name = wz_path.stem
            if name in mounted or name.lower() in ("list", "base"):
                continue
            self._mount_lazy(self.root.add(name), single_file=wz_path, top=True)
            mounted.add(name)
        for top in self._ms_entries:
            if top not in mounted:
                self._mount_lazy(self.root.add(top), top=True)

    def _mount_lazy(
        self, node: WzNode, folder: Path | None = None, single_file: Path | None = None, top: bool = False
    ) -> None:
        def load(target: WzNode) -> None:
            if folder is not None:
                self._load_folder(target, folder)
            if single_file is not None:
                self._load_file(target, single_file)
            if top:
                self._overlay_ms(target)

        node.set_lazy(load)

    def _load_file(self, node: WzNode, path: Path) -> None:
        try:
            wzf = WzFile(path, self._key_hint, self._version_hint)
        except ValueError as exc:
            logger.warning("Skipping %s: %s", path, exc)
            return
        self._open_files.append(wzf)
        self._add_entries(node, wzf, wzf.entries)

    def _load_folder(self, node: WzNode, folder: Path) -> None:
        for path in package_files(folder):
            self._load_file(node, path)
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            if _is_package_folder(sub):
                self._mount_lazy(node.add(sub.name), folder=sub)

    def _add_entries(self, node: WzNode, wzf: WzFile, entries: list[WzEntry]) -> None:
        for entry in entries:
            child = node.add(entry.name)
            if entry.is_dir:
                self._add_entries(child, wzf, entry.children)
            else:
                child.set_image(wzf.make_image(entry))

    def _overlay_ms(self, top_node: WzNode) -> None:
        for ms, entry in self._ms_entries.get(top_node.name, []):
            parts = entry.name.split("/")[1:]
            node = top_node
            for part in parts[:-1]:
                node = node.add(part)
            node.add(parts[-1]).set_image(ms.make_image(entry))

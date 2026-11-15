"""The `WzNode` tree that every package, directory, image, and property is exposed through."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from wz.values import WzCanvas, WzUol

if TYPE_CHECKING:
    from wz.wz_image import WzImage

_MAX_LINK_DEPTH = 16
_CANVAS_LINK_KEYS = ("_inlink", "_outlink", "source")


class WzNode:
    """A named node in the virtual WZ tree.

    Directories, `.img` files, and properties are all nodes. An `.img` node carries a `WzImage`
    and parses its children the first time they are accessed.
    """

    __slots__ = ("_children", "_image", "_lazy", "_loaded", "name", "parent", "value")

    def __init__(self, name: str, parent: WzNode | None = None, value: Any = None) -> None:
        self.name = name
        self.parent = parent
        self.value = value
        self._children: dict[str, WzNode] = {}
        self._image: WzImage | None = None
        self._lazy: Callable[[WzNode], None] | None = None
        self._loaded = True

    # Tree construction --------------------------------------------------------------------

    def add(self, name: str) -> WzNode:
        """Return the child called `name`, creating it if needed."""
        child = self._children.get(name)
        if child is None:
            child = WzNode(name, self)
            self._children[name] = child
        return child

    def attach(self, child: WzNode) -> WzNode:
        """Insert `child` under this node, replacing any existing child with the same name."""
        child.parent = self
        self._children[child.name] = child
        return child

    def set_image(self, image: WzImage) -> None:
        """Mark this node as an `.img` node whose children are parsed lazily from `image`."""
        self._image = image
        self._loaded = False
        self._children = {}

    def set_lazy(self, loader: Callable[[WzNode], None]) -> None:
        """Populate this (directory) node's children by calling `loader(self)` on first access."""
        self._lazy = loader
        self._loaded = False

    def unload(self) -> None:
        """Drop parsed children of this `.img` node (and its cached bytes) to free memory."""
        if self._image is not None:
            self._children = {}
            self._loaded = False
            self._image.release()

    # Accessors ----------------------------------------------------------------------------

    @property
    def image(self) -> WzImage | None:
        """The `WzImage` backing this node, if it is an `.img` node."""
        return self._image

    @property
    def is_image(self) -> bool:
        """Whether this node is an `.img` file."""
        return self._image is not None

    @property
    def children(self) -> dict[str, WzNode]:
        """Children keyed by name, in file order."""
        if not self._loaded:
            self._loaded = True
            if self._image is not None:
                try:
                    self._image.extract(self)
                except Exception:
                    self._loaded = False
                    self._children = {}
                    raise
            elif self._lazy is not None:
                loader, self._lazy = self._lazy, None
                loader(self)
        return self._children

    def __iter__(self) -> Iterator[WzNode]:
        return iter(list(self.children.values()))

    def __len__(self) -> int:
        return len(self.children)

    def __contains__(self, name: str) -> bool:
        return name in self.children

    def __getitem__(self, name: str) -> WzNode:
        return self.children[name]

    def __repr__(self) -> str:
        return f"WzNode({self.path!r}, value={self.value!r})"

    def get(self, name: str) -> WzNode | None:
        """Return the child called `name`, or `None`."""
        return self.children.get(name)

    @property
    def root(self) -> WzNode:
        """The topmost ancestor."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    @property
    def path(self) -> str:
        """Slash-separated path from the root (the root's own name is omitted)."""
        parts: list[str] = []
        node: WzNode | None = self
        while node is not None and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return "/".join(reversed(parts))

    @property
    def image_root(self) -> WzNode | None:
        """The nearest ancestor (inclusive) that is an `.img` node."""
        node: WzNode | None = self
        while node is not None and not node.is_image:
            node = node.parent
        return node

    def find(self, path: str) -> WzNode | None:
        """Find a descendant by slash-separated `path`; `..` and `.` are supported.

        Parameters
        ----------
        path
            Path relative to this node, e.g. `stand/0` or `../move/1`.

        Returns
        -------
        The node at `path`, or `None` if any segment is missing.
        """
        node: WzNode | None = self
        for part in path.replace("\\", "/").split("/"):
            if node is None:
                return None
            if part in ("", "."):
                continue
            if part == "..":
                node = node.parent
                continue
            node = node.children.get(part)
        return node

    # Link resolution ----------------------------------------------------------------------

    def resolve(self) -> WzNode:
        """Follow `UOL` links until a concrete node is reached (or the link is broken)."""
        node = self
        for _ in range(_MAX_LINK_DEPTH):
            if not isinstance(node.value, WzUol) or node.parent is None:
                return node
            target = node.parent.find(node.value.path)
            if target is None:
                return node
            node = target
        return node

    def resolve_canvas(self) -> WzNode:
        """Follow `UOL` links and canvas `_inlink` / `_outlink` / `source` references.

        Returns
        -------
        The node holding the actual bitmap, or the last node reached if a link is broken.
        """
        node = self.resolve()
        for _ in range(_MAX_LINK_DEPTH):
            target = node._canvas_link_target()
            if target is None or target is node:
                return node
            node = target.resolve()
        return node

    def _canvas_link_target(self) -> WzNode | None:
        for key in _CANVAS_LINK_KEYS:
            link = self.children.get(key)
            if link is None or not isinstance(link.value, str) or not link.value:
                continue
            if key == "_inlink":
                img = self.image_root
                return img.find(link.value) if img is not None else None
            return self.root.find(link.value)
        return None

    # Typed value helpers ------------------------------------------------------------------

    def get_value(self, name: str, default: Any = None) -> Any:
        """Return the (UOL-resolved) value of child `name`, or `default` if it is missing."""
        child = self.find(name)
        if child is None:
            return default
        child = child.resolve()
        return default if child.value is None else child.value

    def get_int(self, name: str, default: int = 0) -> int:
        """Return child `name` as an int (strings are parsed), or `default`."""
        value = self.get_value(name)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    def get_str(self, name: str, default: str | None = None) -> str | None:
        """Return child `name` as a string, or `default`."""
        value = self.get_value(name)
        if value is None:
            return default
        return value if isinstance(value, str) else str(value)

    @property
    def canvas(self) -> WzCanvas | None:
        """The bitmap of this node after following links, if any."""
        value = self.resolve_canvas().value
        return value if isinstance(value, WzCanvas) else None

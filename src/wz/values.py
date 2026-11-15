"""Value types stored on `WzNode.value` for non-primitive WZ properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from wz.canvas import decode_canvas

if TYPE_CHECKING:
    from PIL import Image

    from wz.wz_image import WzImage


@dataclass(frozen=True)
class WzVector:
    """A 2D integer vector (`Shape2D#Vector2D`)."""

    x: int
    """Horizontal component."""

    y: int
    """Vertical component."""


@dataclass(frozen=True)
class WzUol:
    """A link to another node, relative to the node holding the link (`UOL`)."""

    path: str
    """Relative path, e.g. `../stand/0`."""


@dataclass(frozen=True)
class WzConvex:
    """A convex polygon (`Shape2D#Convex2D`)."""

    points: tuple[WzVector, ...]
    """Polygon vertices."""


@dataclass
class WzCanvas:
    """A bitmap (`Canvas`). Pixels are decoded lazily with `to_image`."""

    width: int
    """Width in pixels."""

    height: int
    """Height in pixels."""

    format: int
    """Raw texture format id (see `wz.canvas.TextureFormat`)."""

    scale: int
    """Scale exponent; the stored data is `width >> scale` by `height >> scale`."""

    pages: int
    """Number of stacked pages (0 or 1 for ordinary bitmaps)."""

    data_offset: int
    """Offset of the compressed pixel data within the owning image buffer."""

    data_length: int
    """Length of the compressed pixel data in bytes."""

    image: WzImage
    """Owning `.img` that holds the pixel data."""

    def to_image(self, page: int = 0) -> Image.Image:
        """Decode the canvas to an RGBA PIL image.

        Parameters
        ----------
        page
            Page index for multi-page canvases.

        Returns
        -------
        The decoded bitmap in RGBA mode.
        """
        return decode_canvas(self, page)


@dataclass(frozen=True)
class WzSound:
    """Sound data (`Sound_DX8`). Only metadata is kept."""

    data_offset: int
    """Offset of the audio payload in the owning image buffer."""

    data_length: int
    """Length of the audio payload."""

    duration_ms: int
    """Duration in milliseconds."""


@dataclass(frozen=True)
class WzRawData:
    """Opaque binary data (`RawData`)."""

    data_offset: int
    """Offset of the payload in the owning image buffer."""

    data_length: int
    """Length of the payload."""


@dataclass(frozen=True)
class WzVideo:
    """Video data (`Canvas#Video`)."""

    data_offset: int
    """Offset of the payload in the owning image buffer."""

    data_length: int
    """Length of the payload."""

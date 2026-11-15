"""Shared pytest setup: puts `src/` on the import path and locates the sample `Data` folder."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """The client `Data` folder (`$WZ_DATA_DIR` or `<project>/Data`); skips the test if absent."""
    path = Path(os.environ.get("WZ_DATA_DIR", _ROOT / "Data"))
    if not path.is_dir():
        pytest.skip(f"WZ data folder not found: {path}")
    return path

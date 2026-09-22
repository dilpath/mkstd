"""Ensure round-trippable HDF5-stored data.

Run with ``pytest test/`` or ``python test/test_hdf5_roundtrip.py``.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mkstd import Hdf5Standard  # noqa: E402
from mkstd.standards.hdfdict import hdfdict  # noqa: E402


class Leaf(BaseModel):
    """A record with every awkward value."""

    name: str
    note: str = ""
    detail: str | None = None
    flag: bool = True
    count: int = 0
    ratio: float = 0.0
    values: list[float] = []
    matrix: list[list[float]] = []
    labels: list[str] = []
    rows: list[dict[str, Any]] = []
    extra: dict[str, Any] = {}


class Root(BaseModel):
    """A root holding leaves by id."""

    version: int = 1
    leaves: dict[str, Leaf] = {}
    by_label: dict[str, Any] = {}


STANDARD = Hdf5Standard(model=Root)


def _fixture() -> Root:
    return Root(
        leaves={
            "a": Leaf(
                name="a",
                note="",
                detail="best −209.59 — µM, non-ASCII on purpose",
                flag=False,
                count=100,
                ratio=float("nan"),
                values=[-1.0, 0.5, 2.0],
                matrix=[[0.1, 0.2], [0.3, 0.4]],
                labels=["x", "y — z", ""],
                rows=[{"k": 1, "s": "one"}, {"k": None, "s": ""}],
                extra={"nested": {"deeper": [1, 2, 3]}, "none": None},
            ),
            "b": Leaf(name="b"),
        },
        by_label={"": "empty key", "a/b": 2, "100%": 3, ".": 4, "..": 5},
    )


def _roundtrip(root: Root, standard: Hdf5Standard = STANDARD) -> Root:
    with tempfile.TemporaryDirectory() as d:
        filename = str(Path(d) / "data.hdf5")
        standard.save_data(data=root, filename=filename)
        return standard.load_data(filename)


def _equal(x: object, y: object) -> bool:
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or x == y
    if isinstance(x, dict) and isinstance(y, dict):
        return set(x) == set(y) and all(_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(
            _equal(u, v) for u, v in zip(x, y, strict=True)
        )
    return x == y and type(x) is type(y)


def test_roundtrip_is_lossless() -> None:
    root = _fixture()
    back = _roundtrip(root)
    assert _equal(root.model_dump(), back.model_dump())


def test_empty_string_is_a_string() -> None:
    back = _roundtrip(_fixture())
    assert back.leaves["a"].note == ""
    assert back.leaves["a"].labels[2] == ""
    assert back.leaves["b"].note == ""


def test_non_ascii_string_loads() -> None:
    back = _roundtrip(_fixture())
    assert back.leaves["a"].detail == "best −209.59 — µM, non-ASCII on purpose"
    assert back.leaves["a"].labels[1] == "y — z"


def test_keys_with_slash_percent_dots_and_empty_survive() -> None:
    back = _roundtrip(_fixture())
    assert back.by_label == {
        "": "empty key",
        "a/b": 2,
        "100%": 3,
        ".": 4,
        "..": 5,
    }


def test_none_bool_and_scalars_keep_their_types() -> None:
    back = _roundtrip(_fixture())
    leaf = back.leaves["a"]
    assert leaf.detail is not None
    assert leaf.extra["none"] is None
    assert leaf.flag is False and isinstance(leaf.flag, bool)
    assert leaf.count == 100 and isinstance(leaf.count, int)
    assert math.isnan(leaf.ratio)
    assert leaf.rows[1] == {"k": None, "s": ""}
    # a plain-python scalar, not a numpy one, reaches an `Any` field
    assert type(leaf.extra["nested"]["deeper"][0]) is int


def test_escaping_can_be_switched_off() -> None:
    plain = Hdf5Standard(model=Root, escape_keys=False)
    root = Root(leaves={"a": Leaf(name="a")})
    back = _roundtrip(root, plain)
    assert back.leaves["a"].name == "a"


def test_escape_key_is_a_bijection_on_awkward_keys() -> None:
    for key in ("", "/", "a/b", "%", "%2F", "%25", "%00", ".", "..", "a.b"):
        escaped = hdfdict.escape_key(key)
        assert "/" not in escaped and escaped not in ("", ".", "..")
        assert hdfdict.unescape_key(escaped) == key


def test_lazy_load_still_decodes_keys() -> None:
    lazy = Hdf5Standard(model=Root, lazy=True)
    back = _roundtrip(_fixture(), lazy)
    assert set(back.by_label) == {"", "a/b", "100%", ".", ".."}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")

"""Generate Obsidian Canvas files programmatically.

A ``.canvas`` file is JSON Canvas 1.0 — an open spec (https://jsoncanvas.org)
that Obsidian reads natively. Being able to *emit* one means the engine can draw
its own map: the research graph, the data pipeline, the strategy lineage tree.

Node ids are derived from a stable key rather than randomly generated, so
regenerating a canvas updates nodes in place instead of scattering duplicates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Side = Literal["top", "right", "bottom", "left"]

# Obsidian's preset palette. Hex strings also work.
RED, ORANGE, YELLOW, GREEN, CYAN, PURPLE = "1", "2", "3", "4", "5", "6"


def _stable_id(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class _Node:
    key: str
    x: int = 0
    y: int = 0
    width: int = 400
    height: int = 200
    color: str | None = None

    @property
    def id(self) -> str:
        return _stable_id(self.key)

    def _base(self) -> dict:
        d = {
            "id": self.id,
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }
        if self.color:
            d["color"] = self.color
        return d

    def to_json(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError


@dataclass
class TextNode(_Node):
    """A free-floating markdown card."""

    text: str = ""

    def to_json(self) -> dict:
        return {**self._base(), "type": "text", "text": self.text}


@dataclass
class FileNode(_Node):
    """An embedded vault note. `file` is vault-relative, e.g. ``10-Research/x.md``."""

    file: str = ""
    subpath: str | None = None  # e.g. "#Findings" to embed one heading

    def to_json(self) -> dict:
        d = {**self._base(), "type": "file", "file": self.file}
        if self.subpath:
            d["subpath"] = self.subpath
        return d


@dataclass
class LinkNode(_Node):
    """An external URL card."""

    url: str = ""

    def to_json(self) -> dict:
        return {**self._base(), "type": "link", "url": self.url}


@dataclass
class GroupNode(_Node):
    """A labelled frame drawn behind other nodes."""

    label: str = ""

    def to_json(self) -> dict:
        return {**self._base(), "type": "group", "label": self.label}


@dataclass
class CanvasEdge:
    from_node: str  # node *key*, not id
    to_node: str
    from_side: Side = "right"
    to_side: Side = "left"
    label: str | None = None
    color: str | None = None

    def to_json(self) -> dict:
        d = {
            "id": _stable_id(f"{self.from_node}->{self.to_node}:{self.label or ''}"),
            "fromNode": _stable_id(self.from_node),
            "fromSide": self.from_side,
            "toNode": _stable_id(self.to_node),
            "toSide": self.to_side,
        }
        if self.label:
            d["label"] = self.label
        if self.color:
            d["color"] = self.color
        return d


@dataclass
class Canvas:
    """Builder for a ``.canvas`` file."""

    nodes: list[_Node] = field(default_factory=list)
    edges: list[CanvasEdge] = field(default_factory=list)

    def add(self, node: _Node) -> _Node:
        self.nodes.append(node)
        return node

    def link(
        self,
        from_key: str,
        to_key: str,
        *,
        label: str | None = None,
        color: str | None = None,
        from_side: Side = "right",
        to_side: Side = "left",
    ) -> None:
        self.edges.append(
            CanvasEdge(from_key, to_key, from_side, to_side, label=label, color=color)
        )

    def column(
        self,
        nodes: list[_Node],
        *,
        x: int,
        y0: int = 0,
        gap: int = 60,
    ) -> list[_Node]:
        """Stack `nodes` vertically at horizontal position `x`, and register them."""
        y = y0
        for n in nodes:
            n.x, n.y = x, y
            y += n.height + gap
            self.add(n)
        return nodes

    def to_json(self) -> dict:
        return {
            "nodes": [n.to_json() for n in self.nodes],
            "edges": [e.to_json() for e in self.edges],
        }

    def save(self, path: Path) -> Path:
        """Write the canvas. Overwrites — canvases are generated artefacts.

        Obsidian rewrites this file when a human drags a node, so treat a
        regeneration as authoritative and keep hand-drawn work on a separate
        canvas.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )
        return path

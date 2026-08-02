"""Writing into the Obsidian vault.

The vault is the interface, not a log dump. Two rules govern everything here:

1. **Humans and the engine share files.** Generated content lives inside a
   marked region; anything you type outside that region survives regeneration.
   See `notes.write_note`.

2. **Structure lives in frontmatter.** Obsidian's native Bases plugin reads YAML
   frontmatter as database columns, so every generated note carries typed
   properties (sharpe, trades, status...) rather than burying numbers in prose.
"""

from finb.vault.canvas import Canvas, CanvasEdge, FileNode, GroupNode, LinkNode, TextNode
from finb.vault.notes import read_frontmatter, write_note

__all__ = [
    "Canvas",
    "CanvasEdge",
    "FileNode",
    "GroupNode",
    "LinkNode",
    "TextNode",
    "read_frontmatter",
    "write_note",
]

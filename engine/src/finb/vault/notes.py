"""Read and write Obsidian markdown notes with YAML frontmatter.

The central concern is *co-authorship*. These notes are meant to be edited by a
human while the engine keeps rewriting parts of them. So writes are surgical:

    ---
    sharpe: 1.42          <- engine-owned frontmatter keys are merged, not replaced
    my_verdict: promising <- your keys are preserved
    ---

    Whatever you wrote here stays.

    <!-- finb:begin -->
    ...only this region is regenerated...
    <!-- finb:end -->

    And whatever you wrote down here stays too.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

BEGIN = "<!-- finb:begin -->"
END = "<!-- finb:end -->"

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_MANAGED_RE = re.compile(
    re.escape(BEGIN) + r".*?" + re.escape(END),
    re.DOTALL,
)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter dict, body). Missing or malformed frontmatter -> {}."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        # A human mid-edit can easily leave invalid YAML. Don't destroy their file.
        return {}, text
    if not isinstance(fm, dict):
        return {}, text
    return fm, text[m.end() :]


def read_frontmatter(path: Path) -> dict[str, Any]:
    """Frontmatter of an existing note, or {} if absent."""
    if not path.exists():
        return {}
    fm, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
    return fm


def _dump_frontmatter(fm: dict[str, Any]) -> str:
    if not fm:
        return ""
    body = yaml.safe_dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return f"---\n{body}---\n"


def write_note(
    path: Path,
    *,
    frontmatter: dict[str, Any] | None = None,
    managed: str = "",
    initial_body: str = "",
    replace_whole_file: bool = False,
) -> Path:
    """Create or surgically update a note.

    Parameters
    ----------
    frontmatter
        Keys to set. Merged over any existing frontmatter, so hand-added
        properties survive. Pass a key with value ``None`` to delete it.
    managed
        Content for the engine-owned region. Written between the marker
        comments. If the file already has markers, only that span is replaced.
    initial_body
        Prose written *once*, on first creation only, outside the managed
        region — a starting point for your own notes. Never rewritten.
    replace_whole_file
        Escape hatch for notes with no human-authored content. Ignores markers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = frontmatter or {}

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    old_fm, old_body = _split_frontmatter(existing)

    # --- frontmatter: merge, honouring explicit deletions -------------------
    new_fm = {**old_fm}
    for k, v in frontmatter.items():
        if v is None:
            new_fm.pop(k, None)
        else:
            new_fm[k] = v

    managed_block = f"{BEGIN}\n{managed.strip()}\n{END}"

    # --- body ---------------------------------------------------------------
    if replace_whole_file or not existing:
        parts = [p for p in (initial_body.strip(), managed_block) if p]
        new_body = "\n\n".join(parts) + "\n"
    elif _MANAGED_RE.search(old_body):
        new_body = _MANAGED_RE.sub(lambda _: managed_block, old_body, count=1)
    else:
        # File exists but predates the markers — append rather than clobber.
        new_body = old_body.rstrip() + "\n\n" + managed_block + "\n"

    path.write_text(_dump_frontmatter(new_fm) + new_body, encoding="utf-8", newline="\n")
    return path


def wikilink(target: str, alias: str | None = None) -> str:
    """``[[Target|alias]]`` — the unit of connection in the vault."""
    return f"[[{target}|{alias}]]" if alias else f"[[{target}]]"

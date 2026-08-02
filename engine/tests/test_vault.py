"""The co-authorship guarantees are the whole point of the vault layer, so they
get tested first: regeneration must never eat something a human typed."""

from __future__ import annotations

import json

from finb.vault import Canvas, FileNode, TextNode
from finb.vault.notes import read_frontmatter, write_note


def test_creates_note_with_frontmatter_and_managed_region(tmp_path):
    p = tmp_path / "s.md"
    write_note(
        p,
        frontmatter={"sharpe": 1.42, "status": "candidate"},
        managed="| metric | value |\n| --- | --- |\n| trades | 91 |",
        initial_body="## My read\n\n",
    )
    text = p.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "sharpe: 1.42" in text
    assert "## My read" in text
    assert "<!-- finb:begin -->" in text and "<!-- finb:end -->" in text


def test_regeneration_preserves_human_edits(tmp_path):
    p = tmp_path / "s.md"
    write_note(p, frontmatter={"sharpe": 1.0}, managed="old numbers", initial_body="## My read\n")

    # Human edits: adds prose above and below, and their own frontmatter key.
    text = p.read_text(encoding="utf-8")
    text = text.replace("## My read\n", "## My read\n\nThis one looks like curve fitting.\n")
    text = text.replace("sharpe: 1.0", "sharpe: 1.0\nmy_verdict: suspicious")
    text += "\n## Follow-ups\n\n- check turnover\n"
    p.write_text(text, encoding="utf-8")

    # Engine regenerates with new numbers.
    write_note(p, frontmatter={"sharpe": 2.5, "trades": 91}, managed="new numbers")

    out = p.read_text(encoding="utf-8")
    assert "This one looks like curve fitting." in out  # prose above survived
    assert "- check turnover" in out                    # prose below survived
    assert "new numbers" in out and "old numbers" not in out
    fm = read_frontmatter(p)
    assert fm["sharpe"] == 2.5          # engine key updated
    assert fm["trades"] == 91           # engine key added
    assert fm["my_verdict"] == "suspicious"  # human key untouched


def test_frontmatter_key_can_be_deleted(tmp_path):
    p = tmp_path / "s.md"
    write_note(p, frontmatter={"a": 1, "b": 2}, managed="x")
    write_note(p, frontmatter={"b": None}, managed="x")
    assert read_frontmatter(p) == {"a": 1}


def test_malformed_frontmatter_does_not_destroy_file(tmp_path):
    p = tmp_path / "s.md"
    p.write_text("---\nthis: is: not: valid: yaml\n---\n\nprecious prose\n", encoding="utf-8")
    write_note(p, frontmatter={"ok": True}, managed="gen")
    assert "precious prose" in p.read_text(encoding="utf-8")


def test_canvas_emits_valid_json_canvas(tmp_path):
    c = Canvas()
    c.column([TextNode(key="a", text="A"), TextNode(key="b", text="B")], x=0)
    c.add(FileNode(key="note", file="10-Research/x.md", x=600, y=0))
    c.link("a", "note", label="feeds")

    out = c.save(tmp_path / "map.canvas")
    doc = json.loads(out.read_text(encoding="utf-8"))

    assert {n["type"] for n in doc["nodes"]} == {"text", "file"}
    for n in doc["nodes"]:
        assert {"id", "x", "y", "width", "height"} <= n.keys()
    assert doc["nodes"][1]["y"] > doc["nodes"][0]["y"]  # column stacked downward

    e = doc["edges"][0]
    ids = {n["id"] for n in doc["nodes"]}
    assert e["fromNode"] in ids and e["toNode"] in ids
    assert e["label"] == "feeds"


def test_canvas_ids_are_stable_across_regeneration(tmp_path):
    def build():
        c = Canvas()
        c.add(TextNode(key="alpha", text="hello"))
        return json.loads(c.save(tmp_path / "m.canvas").read_text(encoding="utf-8"))

    assert build()["nodes"][0]["id"] == build()["nodes"][0]["id"]

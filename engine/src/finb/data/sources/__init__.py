"""Adapters that turn a vendor's response into `finb.data.lake.BAR_SCHEMA`.

Sources adapt to the canonical schema. Nothing downstream adapts to a source —
otherwise a vendor's field naming leaks into the feature code and swapping
providers becomes a rewrite.
"""

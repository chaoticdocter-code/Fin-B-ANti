"""The data layer: a local Parquet lake with a DuckDB query surface.

No server, no Docker, no cloud. Bars land as Parquet partitioned by symbol and
year; DuckDB reads across partitions with a glob and does the heavy lifting.

The one non-obvious requirement is **gap awareness**. A missing hour of data is
not a neutral absence — it silently becomes a feature ("nothing happened") and
the model learns from a hole. So the lake can always answer *which timestamps it
should have and does not*, and the training pipeline is expected to ask.
"""

from finb.data.lake import BarLake, Timeframe

__all__ = ["BarLake", "Timeframe"]

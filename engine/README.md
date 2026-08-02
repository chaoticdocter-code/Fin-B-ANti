# finb — engine

The Python engine behind the Fin B strategy farm.

This directory is **code only**. All human-facing notes, maps, decisions, and dashboards
live in the Obsidian vault at the repository root (`00-Map`, `10-Research`, …).

Nothing in here places real-money orders. Execution is paper/simulated by construction —
see `src/finb/execution/` and the `FINB_ALLOW_LIVE` guard.

```bash
cd "D:\Fin B\engine"
uv sync
uv run finb --help
```

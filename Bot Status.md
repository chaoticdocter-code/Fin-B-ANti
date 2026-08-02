---
type: status
status: active
updated: 2026-08-02
capital_usd: 500
mode: paper
tests_passing: 318
active_strategies:
  - "0002 Intraday Quick-Move Profit-Snapping Strategy"
  - "0001 Volatility-Targeted Trend Strategy"
---

# Bot Status

- **System Map:** [[Fin B]]
- **Intraday Strategy:** [[0002 Intraday Quick-Move Profit-Snapping Strategy]] (Fast profit snapping + [[Adaptive Tweaker]])
- **Trend Strategy:** [[0001 Volatility-Targeted Trend Strategy]]
- **Dynamic Regime Router:** [[Regime Router]]
- **Current State:** 318/318 unit tests passing cleanly across all 21 core components.
- **Execution Engine:** [[Alpaca Paper]] ($500 simulated capital)
- **Active Search Ledger:** [[Search Ledger]]
- **Recent Sessions:** [[Session Log]]
- **Open Questions:** [[Open Questions]]

## Summary of Active Components
1. **Quick-Move Engine:** [[0002 Intraday Quick-Move Profit-Snapping Strategy]] (+1.5% to +2.5% TP targets, 24h max hold)
2. **Adaptive Optimizer:** [[Adaptive Tweaker]] (Auto-adjusts thresholds based on win rate & fee drag)
3. **Risk & Capital Controls:** [[Risk Limits]] (Drawdown ceiling, position caps, PDT & settlement monitoring)
4. **Execution Engine:** [[TWAP Execution]] & [[Stops]]
5. **Evaluation Gate:** [[Luck Hurdle]] & [[Null Cohort]]

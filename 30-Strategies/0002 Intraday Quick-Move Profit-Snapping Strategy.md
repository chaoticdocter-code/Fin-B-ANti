---
type: strategy
status: active
archetype: QuickMove
capital_usd: 500
mode: paper
take_profit_pct: 0.02
stop_loss_pct: 0.012
max_hold_hours: 24.0
updated: 2026-08-02
---

# 0002 — Intraday Quick-Move Profit-Snapping Strategy

The **Intraday Quick-Move Strategy** is engineered specifically for fast capital rotation and rapid profit capture on liquid crypto majors.

## Strategy Mechanics

1. **Liquid Majors Universe:**
   - Active on `BTC/USD`, `ETH/USD`, `SOL/USD`, and `AVAX/USD` (lowest venue transaction fees).

2. **Rapid Profit Snapping:**
   - Target profit lock-in at **+1.5% to +2.5%** above entry price.
   - Position closes immediately upon reaching profit target, freeing capital to rotate into the next emerging signal.

3. **Tight Risk Management:**
   - 1.0 ATR / -1.2% tight stop loss.
   - Max 24-hour hold time (eliminates multi-week capital lockup).

4. **Adaptive Feedback Loop ([[Adaptive Tweaker]]):**
   - Automatically monitors win rate, fee drag, and trade outcomes.
   - Dynamically tweaks entry thresholds ($RVOL$), profit targets, and stop loss distances based on live trade performance.

---

## System Integration Links
- **Engine Specification:** [[QuickMove Engine]] (`finb.models.quick_move`)
- **Optimization Loop:** [[Adaptive Tweaker]] (`finb.evaluation.tweaker`)
- **System Map:** [[Fin B]]
- **Search Ledger:** [[Search Ledger]]
- **Daily Session Log:** [[Session Log]]

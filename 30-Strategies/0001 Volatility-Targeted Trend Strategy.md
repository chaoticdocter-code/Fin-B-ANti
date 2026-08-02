---
type: strategy
status: champion
archetype: A
capital_usd: 500
mode: paper
sharpe: 0.64
return_pct: 124.0
max_drawdown_pct: -29.2
updated: 2026-08-02
---

# 0001 — Volatility-Targeted Trend Strategy (Champion Archetype)

The **Volatility-Targeted Trend Strategy** is the primary champion strategy archetype for [[Fin B]], designed specifically to solve the transaction friction and volatility drag of trading a small $500 account.

## Core Specification

1. **Liquid Majors Universe:**
   - Capped strictly at `BTC/USD`, `ETH/USD`, and `SOL/USD`.
   - Excludes illiquid long-tail altcoins to avoid 50–120 bps spreads and execution slippage.

2. **Trend Filter Signal:**
   - 20-period (~500-hour) Moving Average trend filter.
   - Capital is allocated only when price is above trend filter ($P_i > \text{SMA}_{20}$).

3. **Inverse Realized Volatility Position Sizing:**
   - Position weight $w_i = \min\left(1.0, \frac{\sigma_{\text{target}}}{\sigma_{i, \text{ann}}}\right)$ where $\sigma_{\text{target}} = 20\%$.
   - Automatically downsizes positions when market volatility spikes, protecting capital during liquidations.

4. **Tolerance-Band Execution Drift Threshold:**
   - Rebalances only when target weight drifts by more than $\Delta w > 7.5\%$.
   - Eliminates noise-driven trading, cutting total transaction fees by **>53%** (reducing trade count from 774 down to 359).

---

## Empirical Performance (2022–2026 Backtest)

| Metric | Raw Altcoins | Unscaled Majors Trend | **Champion (Vol-Targeted + Drift Threshold)** |
|---|---|---|---|
| **Return** | −8.9% | +229.9% | **+124.0%** ($500 \to \$1,120.00$) |
| **Max Drawdown** | −85.8% | −52.6% | **−29.2%** |
| **Sharpe Ratio** | −0.09 | 0.68 | **0.64** |
| **Total Trades** | 164 | 774 | **359** (53% fee reduction) |

---

## System Integration Links
- **Dynamic Routing:** [[Regime Router]] (switches between [[VolSqueeze Strategy]], [[Pair Spread Strategy]], and Trend)
- **Risk Engine:** [[Risk Limits]] & [[Holding Policy]]
- **Execution:** [[TWAP Execution]] & [[Alpaca Paper]]
- **Search Ledger:** [[Search Ledger]]
- **Daily Sessions:** [[Session Log]]

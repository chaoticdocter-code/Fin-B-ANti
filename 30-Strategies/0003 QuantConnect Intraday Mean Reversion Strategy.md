---
type: strategy
status: active
archetype: QCMeanReversion
capital_usd: 500
mode: paper
bb_period: 20
rsi_period: 14
updated: 2026-08-02
---

# 0003 — QuantConnect Intraday Mean Reversion Strategy

The **QuantConnect Intraday Mean Reversion Strategy** is an institutional strategy archetype ingested from QuantConnect's open-source LEAN algorithm library.

## Strategy Mechanics

1. **Target Universe:**
   - Active on `BTC/USD`, `ETH/USD`, `SOL/USD`, and `AVAX/USD`.

2. **Signal Indicators:**
   - **Bollinger Band %B Indicator:** $\%B = \frac{P - \text{BB}_{\text{lower}}}{\text{BB}_{\text{upper}} - \text{BB}_{\text{lower}}}$
   - **RSI Oscillator:** 14-period RSI.

3. **Entry Triggers:**
   - **LONG Capitulation Bounce (`BUY`):** $\%B < 0.05$ & $\text{RSI} < 35.0$. Target: Mean reversion to 20-period Moving Average ($\text{SMA}_{20}$).
   - **SHORT Climax Pullback (`SELL`):** $\%B > 0.95$ & $\text{RSI} > 65.0$. Target: Mean reversion to 20-period Moving Average ($\text{SMA}_{20}$).

4. **Risk Management:**
   - 1.5x ATR trailing stop loss.

---

## System Integration Links
- **Engine Module:** [[QC Mean Reversion Engine]] (`finb.models.qc_mean_reversion`)
- **LEAN Exporter:** [[QC Adaptor]] (`finb.sim.qc_adaptor`)
- **Dynamic Regime Router:** [[Regime Router]]
- **System Map:** [[Fin B]]
- **Search Ledger:** [[Search Ledger]]

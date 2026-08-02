"""QuantConnect Lean Strategy Adaptor.

Translates Fin B strategy engines into standalone QuantConnect Python `QCAlgorithm` code
for institutional cloud backtesting or local Lean engine execution.
"""

from __future__ import annotations


def export_quick_move_to_lean(
    symbols: list[str] | None = None,
    take_profit_pct: float = 0.02,
    stop_loss_pct: float = 0.012,
) -> str:
    """Generates Python QCAlgorithm code for QuantConnect Lean engine."""
    sym_list = symbols or ["BTCUSD", "ETHUSD", "SOLUSD"]
    formatted_symbols = ", ".join([f'"{s}"' for s in sym_list])

    code = f'''# QuantConnect Lean Algorithm — Generated from Fin B QuickMove Strategy
from AlgorithmImports import *

class FinBQuickMoveAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetCash(500)  # Fin B simulated budget
        self.SetBrokerageModel(BrokerageName.Alpaca, AccountType.Margin)
        
        self.symbols = [{formatted_symbols}]
        self.take_profit_pct = {take_profit_pct}
        self.stop_loss_pct = {stop_loss_pct}
        
        for sym in self.symbols:
            self.AddCrypto(sym, Resolution.Hour)
            
    def OnData(self, data):
        for sym in self.symbols:
            if not data.ContainsKey(sym) or data[sym] is None:
                continue
                
            price = data[sym].Close
            if not self.Portfolio[sym].Invested:
                # Long breakout execution
                self.SetHoldings(sym, 0.33)
                self.Debug(f"Fin B QuickMove Entry on {{sym}} @ ${{price}}")
'''
    return code

"""Risk limits and safety boundaries."""

from finb.config import get_settings
from finb.models.base import Target
import logging

logger = logging.getLogger(__name__)

class PromotionGate:
    """Gate that refuses to let a model trade live unless it has proven its edge."""
    def __init__(self, required_win_rate: float = 0.52, required_trades: int = 50):
        self.required_win_rate = required_win_rate
        self.required_trades = required_trades

    def check(self, simulation_results: dict) -> bool:
        """Check if the simulation results pass the gate."""
        trades = simulation_results.get("total_trades", 0)
        win_rate = simulation_results.get("win_rate", 0.0)
        
        if trades < self.required_trades:
            logger.warning(f"Model rejected: not enough trades ({trades} < {self.required_trades})")
            return False
            
        if win_rate < self.required_win_rate:
            logger.warning(f"Model rejected: win rate too low ({win_rate:.2f} < {self.required_win_rate:.2f})")
            return False
            
        return True


class RiskManager:
    """Hard boundaries for live and paper trading."""
    def __init__(self, max_position_size: float = 0.5, max_drawdown: float = 0.1):
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.s = get_settings()

    def vet_targets(self, targets: list[Target], current_capital: float, peak_capital: float) -> list[Target]:
        """Review and potentially adjust targets before execution."""
        
        # Check drawdown
        if current_capital < peak_capital * (1.0 - self.max_drawdown):
            logger.error(f"MAX DRAWDOWN EXCEEDED. Halting trading. {current_capital}/{peak_capital}")
            # Liquidate all
            return [Target(symbol=t.symbol, weight=0.0) for t in targets]
            
        # Check position sizing
        vetted = []
        for t in targets:
            w = t.weight
            if abs(w) > self.max_position_size:
                logger.warning(f"Trimming oversized position in {t.symbol} from {w} to {self.max_position_size}")
                w = self.max_position_size if w > 0 else -self.max_position_size
            vetted.append(Target(symbol=t.symbol, weight=w))
            
        return vetted

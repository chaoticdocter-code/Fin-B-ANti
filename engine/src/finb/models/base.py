"""Base interface for all trading models."""

from abc import ABC, abstractmethod
import polars as pl
from dataclasses import dataclass

@dataclass
class Target:
    """The desired target position for a symbol."""
    symbol: str
    weight: float  # Target weight [-1.0, 1.0]
    
class Model(ABC):
    """A trading model takes features and emits position targets."""
    
    @abstractmethod
    def fit(self, X: pl.DataFrame, y: pl.Series) -> None:
        """Train the model if it has parameters to fit."""
        pass
        
    @abstractmethod
    def predict(self, df: pl.DataFrame) -> list[Target]:
        """Given a dataframe of current features, return a list of Targets."""
        pass

"""Gradient-boosted trees with purged cross-validation.

LightGBM rather than a neural network, on the research's evidence: on the Qlib
benchmarks, Transformer and TabNet produce *negative* annualised returns on raw
features, while gradient boosting is the consistent performer on tabular
financial data. Time-series foundation models fare worse still — negative
out-of-sample R² on return direction, i.e. worse than predicting zero.

The scoring here is deliberately not accuracy. Accuracy on a 55/45 class split
is dominated by the majority class and tells you nothing tradeable. What matters
is whether the model's *ranking* carries information, so we report AUC and the
information coefficient, and then convert predictions into a P&L that pays
costs — because a model can rank well and still lose money at 57bps a round trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from finb.models.cv import PurgedKFold, leakage_report

DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 60,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbosity": -1,
    "n_estimators": 300,
}
"""Deliberately small and heavily regularised. With a few thousand samples and
a signal-to-noise ratio near zero, capacity is not the constraint — it is the
enemy."""


@dataclass
class TrainResult:
    oof: np.ndarray
    """Out-of-fold predicted probabilities. The only predictions worth scoring."""

    fold_auc: list[float] = field(default_factory=list)
    fold_n_train: list[int] = field(default_factory=list)
    importance: dict[str, float] = field(default_factory=dict)
    leakage: dict = field(default_factory=dict)
    models: list = field(default_factory=list)

    @property
    def mean_auc(self) -> float:
        return float(np.mean(self.fold_auc)) if self.fold_auc else 0.5

    def information_coefficient(self, y: np.ndarray) -> float:
        """Spearman correlation between prediction and outcome.

        Around 0.03 is a realistic ceiling for a retail cross-sectional system;
        anything much above 0.10 on financial data should be treated as a bug
        report rather than a result.
        """
        mask = np.isfinite(self.oof)
        if mask.sum() < 10:
            return 0.0
        from scipy.stats import spearmanr

        rho, _ = spearmanr(self.oof[mask], y[mask])
        return float(rho) if np.isfinite(rho) else 0.0

    def summary(self, y: np.ndarray) -> str:
        return (
            f"AUC {self.mean_auc:.4f} (folds: "
            f"{', '.join(f'{a:.3f}' for a in self.fold_auc)}), "
            f"IC {self.information_coefficient(y):+.4f}, "
            f"purged CV dropped {self.leakage.get('dropped_fraction', 0):.1%} of training rows"
        )


def train_purged(
    X: np.ndarray,
    y: np.ndarray,
    t1_idx: np.ndarray,
    *,
    n_splits: int = 5,
    embargo_pct: float = 0.02,
    params: dict | None = None,
    feature_names: list[str] | None = None,
) -> TrainResult:
    """Train with purged, embargoed CV and return out-of-fold predictions.

    Every score here is out-of-fold by construction. In-sample scores on
    financial data are not weak evidence, they are no evidence.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    p = {**DEFAULT_PARAMS, **(params or {})}
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)

    result = TrainResult(
        oof=np.full(y.size, np.nan),
        leakage=leakage_report(t1_idx, n_splits=n_splits, embargo_pct=embargo_pct),
    )
    gains: dict[str, float] = {}

    for train_idx, test_idx in cv.split(t1_idx):
        if train_idx.size < 100 or len(np.unique(y[train_idx])) < 2:
            continue

        model = lgb.LGBMClassifier(**p)
        model.fit(X[train_idx], y[train_idx])

        proba = model.predict_proba(X[test_idx])[:, 1]
        result.oof[test_idx] = proba
        result.models.append(model)
        result.fold_n_train.append(int(train_idx.size))

        if len(np.unique(y[test_idx])) > 1:
            result.fold_auc.append(float(roc_auc_score(y[test_idx], proba)))

        if feature_names is not None:
            for name, imp in zip(feature_names, model.feature_importances_, strict=True):
                gains[name] = gains.get(name, 0.0) + float(imp)

    total = sum(gains.values()) or 1.0
    result.importance = {k: v / total for k, v in sorted(gains.items(), key=lambda kv: -kv[1])}
    return result


def signal_to_returns(
    oof: np.ndarray,
    ret: np.ndarray,
    *,
    threshold: float = 0.5,
    cost_bps: float = 57.0,
) -> np.ndarray:
    """Convert predictions into per-trade returns net of a round trip.

    Long-only: take the trade when the predicted probability clears `threshold`,
    stand aside otherwise. `cost_bps` is charged on every taken trade — which is
    what usually turns a promising IC into a losing strategy.
    """
    take = np.isfinite(oof) & (oof > threshold)
    out = np.where(take, ret - cost_bps / 1e4, 0.0)
    return out[np.isfinite(oof)]

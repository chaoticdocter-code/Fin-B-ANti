"""Models, and the cross-validation that keeps their scores honest.

Ordinary k-fold is wrong on financial data and wrong in the flattering
direction. Labels built over a horizon overlap each other, so a neighbouring
sample in the training set carries the answer to a test sample. The model does
not need to generalise; it only needs to remember. `cv.PurgedKFold` removes
those neighbours.
"""

from finb.models.cv import PurgedKFold, leakage_report

__all__ = ["PurgedKFold", "leakage_report"]

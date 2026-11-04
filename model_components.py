"""Estimator components that must stay importable when saved Pipelines are loaded.

They live outside ``train_model.py`` because pickles of classes defined in a script
run as ``__main__`` cannot be loaded by the Streamlit app.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier


class FoldBalancedXGBClassifier(XGBClassifier):
    """XGBClassifier whose ``scale_pos_weight`` is computed from the labels given to ``fit``.

    Inside cross-validation ``fit`` only sees the training fold, so the class ratio never
    uses validation-fold or test labels. ``scale_pos_weight`` is overwritten on every fit
    and therefore reflects the most recent training labels.
    """

    def fit(self, X, y, **kwargs):
        labels = np.asarray(y)
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        if positives == 0 or negatives == 0:
            raise ValueError('Both classes (0 and 1) are required to compute scale_pos_weight.')
        self.set_params(scale_pos_weight=negatives / positives)
        return super().fit(X, y, **kwargs)


class PotableThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Apply a fixed decision threshold to the class-1 (Potable) probability.

    ``threshold`` is chosen elsewhere, from out-of-fold *training* predictions only.
    A sample is predicted Potable only if P(Potable) >= threshold, which makes false
    "Potable" predictions rarer at the cost of more false "Not Potable" predictions.
    ``threshold=None`` keeps the wrapped model's native decision rule.
    Probabilities are passed through unchanged.
    """

    def __init__(self, estimator, threshold=None):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        if self.threshold is None:
            return self.estimator_.predict(X)
        proba = self.estimator_.predict_proba(X)
        potable = proba[:, list(self.classes_).index(1)]
        return np.where(potable >= self.threshold, 1, 0)

    # Expose the wrapped model's explanations so the app's importance chart keeps working.
    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_

    @property
    def coef_(self):
        return self.estimator_.coef_

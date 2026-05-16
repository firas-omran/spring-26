from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class OOBRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Random Forest classifier with explicit OOB accounting.

    The ensemble logic is implemented here: bootstrap sampling, OOB prediction
    aggregation, voting, and OOB permutation importance. The base learner is
    sklearn's DecisionTreeClassifier, as allowed by the lab statement.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: str | float | int | None = "sqrt",
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
        class_weight: str | dict | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X_df = self._as_frame(X)
        y_arr = np.asarray(y)
        n_samples = X_df.shape[0]

        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]
        self.classes_ = np.sort(np.unique(y_arr))
        self.n_classes_ = len(self.classes_)

        self.estimators_ = []
        self.bootstrap_indices_ = []
        self.oob_indices_ = []

        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            if self.bootstrap:
                sample_idx = rng.integers(0, n_samples, size=n_samples)
            else:
                sample_idx = np.arange(n_samples)

            used = np.zeros(n_samples, dtype=bool)
            used[sample_idx] = True
            oob_idx = np.flatnonzero(~used)

            tree = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                random_state=tree_seed,
            )
            tree.fit(X_df.iloc[sample_idx], y_arr[sample_idx])

            self.estimators_.append(tree)
            self.bootstrap_indices_.append(sample_idx)
            self.oob_indices_.append(oob_idx)

        oob_pred, oob_proba, oob_mask = self._collect_oob_predictions(X_df)
        self.oob_mask_ = oob_mask
        self.oob_decision_function_ = oob_proba
        self.oob_prediction_ = oob_pred
        self.oob_score_ = accuracy_score(y_arr[oob_mask], oob_pred[oob_mask]) if oob_mask.any() else np.nan
        self.feature_importances_ = self._mean_impurity_importance()
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X_df = self._as_frame(X)
        proba_sum = np.zeros((len(X_df), self.n_classes_), dtype=float)
        for tree in self.estimators_:
            proba_sum += self._aligned_proba(tree, X_df)
        return proba_sum / len(self.estimators_)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def oob_permutation_importance(
        self,
        X,
        y,
        n_repeats: int = 7,
        random_state: int | None = None,
    ) -> Dict[str, float]:
        """Estimate feature importance as OOB accuracy drop after permutation."""
        X_df = self._as_frame(X)
        y_arr = np.asarray(y)
        baseline_pred, _, baseline_mask = self._collect_oob_predictions(X_df)
        baseline = accuracy_score(y_arr[baseline_mask], baseline_pred[baseline_mask])
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        importances: Dict[str, float] = {}
        for feature in X_df.columns:
            drops = []
            for _ in range(n_repeats):
                permuted_values = X_df[feature].to_numpy(copy=True)
                rng.shuffle(permuted_values)
                pred, _, mask = self._collect_oob_predictions(
                    X_df,
                    permuted_feature=feature,
                    permuted_values=permuted_values,
                )
                permuted_score = accuracy_score(y_arr[mask], pred[mask])
                drops.append(baseline - permuted_score)
            importances[str(feature)] = float(np.mean(drops))
        return importances

    def _collect_oob_predictions(self, X_df, permuted_feature=None, permuted_values=None):
        X_df = self._as_frame(X_df).copy()
        n_samples = X_df.shape[0]
        proba_sum = np.zeros((n_samples, self.n_classes_), dtype=float)
        counts = np.zeros(n_samples, dtype=int)

        for tree, oob_idx in zip(self.estimators_, self.oob_indices_):
            if len(oob_idx) == 0:
                continue
            X_oob = X_df.iloc[oob_idx].copy()
            if permuted_feature is not None:
                X_oob.loc[:, permuted_feature] = permuted_values[oob_idx]
            proba_sum[oob_idx] += self._aligned_proba(tree, X_oob)
            counts[oob_idx] += 1

        mask = counts > 0
        safe_counts = counts.copy()
        safe_counts[safe_counts == 0] = 1
        oob_proba = proba_sum / safe_counts[:, None]
        oob_pred = self.classes_[np.argmax(oob_proba, axis=1)]
        return oob_pred, oob_proba, mask

    def _aligned_proba(self, tree: DecisionTreeClassifier, X_df: pd.DataFrame):
        raw = tree.predict_proba(X_df)
        aligned = np.zeros((len(X_df), self.n_classes_), dtype=float)
        for local_idx, label in enumerate(tree.classes_):
            global_idx = np.where(self.classes_ == label)[0][0]
            aligned[:, global_idx] = raw[:, local_idx]
        return aligned

    def _mean_impurity_importance(self):
        values = np.zeros(self.n_features_in_, dtype=float)
        for tree in self.estimators_:
            values += tree.feature_importances_
        values /= len(self.estimators_)
        return values

    def _as_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        if hasattr(self, "feature_names_in_") and np.asarray(X).shape[1] == len(self.feature_names_in_):
            return pd.DataFrame(X, columns=self.feature_names_in_)
        return pd.DataFrame(X)


def oob_scorer(estimator, X, y):
    return estimator.oob_score_

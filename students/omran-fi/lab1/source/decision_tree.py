import numpy as np
import pandas as pd

from .criteria import gini


class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        proba=None,
        n_samples=0.0,
    ):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

        self.value = value
        self.proba = proba
        self.n_samples = n_samples

        self.is_leaf = left is None and right is None

        # Probabilities used to route missing values to child branches.
        self.left_prob = 0.5
        self.right_prob = 0.5


class DecisionTree:

    def __init__(self, max_depth=10, min_samples_split=2):

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.root = None
        self.n_features = None
        self.feature_importances_ = None

    def fit(self, X, y):

        X = self._to_numpy_X(X)
        y = self._to_numpy_y(y)
        sample_weight = np.ones(len(y), dtype=float)

        self.n_features = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features, dtype=float)

        self.root = self._build_tree(X, y, sample_weight, depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def _build_tree(self, X, y, sample_weight, depth):

        total_weight = float(sample_weight.sum())
        majority = self._majority(y, sample_weight)
        proba = self._positive_proba(y, sample_weight)

        if (
            depth >= self.max_depth
            or total_weight < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return Node(
                value=majority,
                proba=proba,
                n_samples=total_weight,
            )

        best_feature, best_threshold, best_gain, left_prob, right_prob = self._best_split(
            X,
            y,
            sample_weight,
        )

        if best_feature is None:
            return Node(
                value=majority,
                proba=proba,
                n_samples=total_weight,
            )

        split = self._partition_dataset(
            X,
            y,
            sample_weight,
            best_feature,
            best_threshold,
            left_prob,
            right_prob,
        )

        left_X, left_y, left_weight, right_X, right_y, right_weight = split

        if left_weight.sum() == 0 or right_weight.sum() == 0:
            return Node(
                value=majority,
                proba=proba,
                n_samples=total_weight,
            )

        self.feature_importances_[best_feature] += best_gain

        left_child = self._build_tree(left_X, left_y, left_weight, depth + 1)
        right_child = self._build_tree(right_X, right_y, right_weight, depth + 1)

        node = Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            value=majority,
            proba=proba,
            n_samples=total_weight,
        )
        node.left_prob = left_prob
        node.right_prob = right_prob

        return node

    def _best_split(self, X, y, sample_weight):

        best_feature = None
        best_threshold = None
        best_gini = np.inf
        best_gain = -np.inf
        best_left_prob = None
        best_right_prob = None

        parent_gini = self._weighted_gini(y, sample_weight)

        for feature in range(X.shape[1]):
            feature_values = X[:, feature]
            missing_mask = pd.isna(feature_values)
            known_mask = ~missing_mask
            known_values = feature_values[known_mask]

            if len(np.unique(known_values)) < 2:
                continue

            thresholds = self._candidate_thresholds(known_values)
            missing_y = y[missing_mask]
            missing_weight = sample_weight[missing_mask]
            missing_class_weight = self._class_weight_sums(missing_y, missing_weight)

            for threshold in thresholds:
                left_known_mask = known_mask & (feature_values <= threshold)
                right_known_mask = known_mask & (feature_values > threshold)

                left_known_weight = float(sample_weight[left_known_mask].sum())
                right_known_weight = float(sample_weight[right_known_mask].sum())

                if left_known_weight == 0 or right_known_weight == 0:
                    continue

                total_known = left_known_weight + right_known_weight
                left_prob = left_known_weight / total_known
                right_prob = right_known_weight / total_known

                left_class_weight = self._class_weight_sums(
                    y[left_known_mask],
                    sample_weight[left_known_mask],
                ) + missing_class_weight * left_prob
                right_class_weight = self._class_weight_sums(
                    y[right_known_mask],
                    sample_weight[right_known_mask],
                ) + missing_class_weight * right_prob

                left_weight = left_class_weight.sum()
                right_weight = right_class_weight.sum()

                if left_weight == 0 or right_weight == 0:
                    continue

                left_gini = self._gini_from_class_weights(left_class_weight)
                right_gini = self._gini_from_class_weights(right_class_weight)
                weighted_gini = (
                    left_weight * left_gini + right_weight * right_gini
                ) / (left_weight + right_weight)

                gain = parent_gini - weighted_gini

                is_better_split = weighted_gini < best_gini - 1e-12
                is_tie_with_better_gain = (
                    np.isclose(weighted_gini, best_gini)
                    and gain > best_gain + 1e-12
                )
                is_full_tie = (
                    np.isclose(weighted_gini, best_gini)
                    and np.isclose(gain, best_gain)
                    and (
                        best_feature is None
                        or feature < best_feature
                        or (
                            feature == best_feature
                            and float(threshold) < float(best_threshold)
                        )
                    )
                )

                if is_better_split or is_tie_with_better_gain or is_full_tie:
                    best_gini = float(weighted_gini)
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_gain = float(gain)
                    best_left_prob = float(left_prob)
                    best_right_prob = float(right_prob)

        return best_feature, best_threshold, best_gain, best_left_prob, best_right_prob

    def _candidate_thresholds(self, values):

        unique_values = np.unique(values.astype(float))

        if len(unique_values) < 2:
            return unique_values

        return (unique_values[:-1] + unique_values[1:]) / 2.0

    def _majority(self, y, sample_weight):

        class_weights = self._class_weight_sums(y, sample_weight)
        return int(np.argmax(class_weights))

    def _positive_proba(self, y, sample_weight):

        total = float(sample_weight.sum())
        if total == 0:
            return 0.0

        return float(sample_weight[y == 1].sum() / total)

    def _class_weight_sums(self, y, sample_weight):

        if len(y) == 0:
            return np.zeros(2, dtype=float)

        n_classes = max(2, int(np.max(y)) + 1)
        return np.bincount(y.astype(int), weights=sample_weight, minlength=n_classes).astype(float)

    def _gini_from_class_weights(self, class_weights):

        total = float(class_weights.sum())
        if total == 0:
            return 0.0

        probabilities = class_weights / total
        return float(1.0 - np.sum(probabilities ** 2))

    def _weighted_gini(self, y, sample_weight):

        if np.allclose(sample_weight, sample_weight[0]):
            return float(gini(y))

        class_weights = self._class_weight_sums(y, sample_weight)
        return self._gini_from_class_weights(class_weights)

    def _partition_dataset(self, X, y, sample_weight, feature, threshold, left_prob, right_prob):

        feature_values = X[:, feature]
        missing_mask = pd.isna(feature_values)
        known_mask = ~missing_mask

        left_known_mask = known_mask & (feature_values <= threshold)
        right_known_mask = known_mask & (feature_values > threshold)

        left_X = X[left_known_mask]
        left_y = y[left_known_mask]
        left_weight = sample_weight[left_known_mask]

        right_X = X[right_known_mask]
        right_y = y[right_known_mask]
        right_weight = sample_weight[right_known_mask]

        if np.any(missing_mask):
            missing_X = X[missing_mask]
            missing_y = y[missing_mask]
            missing_weight = sample_weight[missing_mask]

            left_X = np.concatenate([left_X, missing_X], axis=0)
            left_y = np.concatenate([left_y, missing_y], axis=0)
            left_weight = np.concatenate([left_weight, missing_weight * left_prob], axis=0)

            right_X = np.concatenate([right_X, missing_X], axis=0)
            right_y = np.concatenate([right_y, missing_y], axis=0)
            right_weight = np.concatenate([right_weight, missing_weight * right_prob], axis=0)

        positive_left = left_weight > 1e-12
        positive_right = right_weight > 1e-12

        return (
            left_X[positive_left],
            left_y[positive_left],
            left_weight[positive_left],
            right_X[positive_right],
            right_y[positive_right],
            right_weight[positive_right],
        )

    def _to_numpy_X(self, X):

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return X

    def _to_numpy_y(self, y):

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = np.asarray(y).reshape(-1)

        return np.asarray(y, dtype=int)

    def _check_is_fitted(self):

        if self.root is None:
            raise ValueError("DecisionTree instance is not fitted yet. Call fit() first.")

    def predict(self, X):

        self._check_is_fitted()
        X = self._to_numpy_X(X)
        return np.array([self._predict_row(row, self.root) for row in X], dtype=int)

    def predict_proba(self, X):

        self._check_is_fitted()
        X = self._to_numpy_X(X)
        return np.array([self._predict_proba_row(row, self.root) for row in X], dtype=float)

    def _predict_row(self, row, node):

        if node.is_leaf:
            return node.value

        value = row[node.feature]

        if np.isnan(value):
            left_pred = self._predict_row(row, node.left)
            right_pred = self._predict_row(row, node.right)
            pred = node.left_prob * left_pred + node.right_prob * right_pred
            return int(round(pred))

        if value <= node.threshold:
            return self._predict_row(row, node.left)

        return self._predict_row(row, node.right)

    def _predict_proba_row(self, row, node):

        if node.is_leaf:
            return node.proba

        value = row[node.feature]

        if np.isnan(value):
            left_proba = self._predict_proba_row(row, node.left)
            right_proba = self._predict_proba_row(row, node.right)
            return node.left_prob * left_proba + node.right_prob * right_proba

        if value <= node.threshold:
            return self._predict_proba_row(row, node.left)

        return self._predict_proba_row(row, node.right)

    def prune(self, X_val, y_val):

        self._check_is_fitted()
        X_val = self._to_numpy_X(X_val)
        y_val = self._to_numpy_y(y_val)
        sample_weight = np.ones(len(y_val), dtype=float)

        self._prune_node(self.root, X_val, y_val, sample_weight)

    def _prune_node(self, node, X_subset, y_subset, sample_weight):

        if node is None or node.is_leaf or len(y_subset) == 0 or sample_weight.sum() == 0:
            return

        split = self._partition_dataset(
            X_subset,
            y_subset,
            sample_weight,
            node.feature,
            node.threshold,
            node.left_prob,
            node.right_prob,
        )
        left_X, left_y, left_weight, right_X, right_y, right_weight = split

        self._prune_node(node.left, left_X, left_y, left_weight)
        self._prune_node(node.right, right_X, right_y, right_weight)

        baseline_predictions = np.array([self._predict_row(row, node) for row in X_subset], dtype=int)
        baseline_accuracy = self._weighted_accuracy(y_subset, baseline_predictions, sample_weight)

        original_state = (
            node.feature,
            node.threshold,
            node.left,
            node.right,
            node.is_leaf,
        )

        node.feature = None
        node.threshold = None
        node.left = None
        node.right = None
        node.is_leaf = True

        pruned_predictions = np.full(len(y_subset), node.value, dtype=int)
        pruned_accuracy = self._weighted_accuracy(y_subset, pruned_predictions, sample_weight)

        if pruned_accuracy < baseline_accuracy:
            (
                node.feature,
                node.threshold,
                node.left,
                node.right,
                node.is_leaf,
            ) = original_state

    def _weighted_accuracy(self, y_true, y_pred, sample_weight):

        total = float(sample_weight.sum())
        if total == 0:
            return 0.0

        correct_weight = sample_weight[y_true == y_pred].sum()
        return float(correct_weight / total)

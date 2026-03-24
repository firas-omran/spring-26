import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def gain_gini(
    y: np.ndarray,
    split_mask: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    if weights is None:
        weights = np.ones(n)
    left_mask = split_mask
    right_mask = ~split_mask
    n_left = np.sum(weights[left_mask])
    n_right = np.sum(weights[right_mask])
    if n_left == 0 or n_right == 0:
        return 0.0
    # Взвешенный Gini
    def wgini(mask: np.ndarray) -> float:
        w = weights[mask]
        y_sub = y[mask]
        if np.sum(w) == 0:
            return 0.0
        classes = np.unique(y_sub)
        return 1.0 - sum(
            (np.sum(w[y_sub == c]) / np.sum(w)) ** 2 for c in classes
        )
    g_left = wgini(left_mask)
    g_right = wgini(right_mask)
    p_left = n_left / (n_left + n_right)
    p_right = n_right / (n_left + n_right)
    g_parent = wgini(np.ones(n, dtype=bool))
    return g_parent - p_left * g_left - p_right * g_right


@dataclass
class Node:
    feature: Optional[str] = None
    threshold: Optional[Any] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    label: Optional[int] = None  # метка для листа
    proba: Optional[Dict[int, float]] = None  # P(class|leaf)
    # Вероятности перехода из родителя в детей (для пропусков при классификации)
    p_left: float = 0.5
    p_right: float = 0.5
    samples: int = 0
    depth: int = 0


class ID3Tree:
    """
    Бинарное решающее дерево ID3 с критерием Джини.
    Обработка пропусков:
    - Обучение: объекты с пропуском по признаку исключаются из Gain; 
      сохраняются вероятности перехода p_left, p_right.
    - Классификация: при пропуске объект распределяется по детям с весами p_left, p_right.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
        holdout_frac: float = 0.2,
        random_state: Optional[int] = None
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.holdout_frac = holdout_frac
        self.random_state = random_state
        self.root_: Optional[Node] = None
        self.feature_names_: List[str] = []
        self.classes_: np.ndarray = np.array([])

    def _holdout_split(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        n_val = int(n * self.holdout_frac)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y[train_idx]
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y[val_idx]
        return X_train, y_train, X_val, y_val

    def _best_split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str],
        sample_weight: Optional[np.ndarray] = None
    ) -> Tuple[Optional[str], Optional[np.ndarray], float, float]:
        best_gain = -1.0
        best_feat = None
        best_mask = None
        best_p_left, best_p_right = 0.5, 0.5

        for feat in features:
            col = X[feat]
            valid = pd.notna(col)
            if valid.sum() < self.min_samples_leaf:
                continue
            # Бинарный признак: 0 — левое поддерево, 1 — правое
            split_mask = (col == 1) & valid
            n_valid = valid.sum()
            n_left = split_mask.sum()
            n_right = n_valid - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            # Gain только по объектам с определённым значением
            idx_valid = np.where(valid)[0]
            y_sub = y[idx_valid]
            mask_sub = split_mask.values[idx_valid]
            if sample_weight is not None:
                w_sub = sample_weight[idx_valid]
            else:
                w_sub = None
            g = gain_gini(y_sub, mask_sub, w_sub)
            if g > best_gain:
                best_gain = g
                best_feat = feat
                best_mask = split_mask.values
                p_left = n_left / n_valid if n_valid > 0 else 0.5
                best_p_left = p_left
                best_p_right = 1.0 - p_left

        return best_feat, best_mask, best_p_left, best_p_right

    def _build(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        features: List[str],
        depth: int,
        sample_weight: Optional[np.ndarray] = None
    ) -> Node:
        node = Node(depth=depth, samples=len(y))
        if depth >= self.max_depth or len(np.unique(y)) <= 1 or len(features) == 0:
            return self._make_leaf(node, y)
        if len(y) < 2 * self.min_samples_leaf:
            return self._make_leaf(node, y)

        feat, mask, p_left, p_right = self._best_split(X, y, features, sample_weight)
        if feat is None or mask is None:
            return self._make_leaf(node, y)

        left_idx = np.where(mask)[0]
        right_idx = np.where((col == 0) & valid)[0]
        if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
            return self._make_leaf(node, y)

        new_features = [f for f in features if f != feat]
        node.feature = feat
        node.p_left = p_left
        node.p_right = p_right

        X_left = X.iloc[left_idx].reset_index(drop=True)
        y_left = y[left_idx]
        X_right = X.iloc[right_idx].reset_index(drop=True)
        y_right = y[right_idx]

        w_left = sample_weight[left_idx] if sample_weight is not None else None
        w_right = sample_weight[right_idx] if sample_weight is not None else None

        node.left = self._build(X_left, y_left, new_features, depth + 1, w_left)
        node.right = self._build(X_right, y_right, new_features, depth + 1, w_right)

        return node

    def _make_leaf(self, node: Node, y: np.ndarray) -> Node:
        unique, counts = np.unique(y, return_counts=True)
        total = counts.sum()
        node.proba = {int(c): cnt / total for c, cnt in zip(unique, counts)}
        node.label = int(unique[np.argmax(counts)])
        return node

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ID3Tree":
        self.classes_ = np.unique(y)
        self.feature_names_ = list(X.columns)
        X_train, y_train, X_val, y_val = self._holdout_split(X, y)
        self.root_ = self._build(X_train, y_train, self.feature_names_, 0)
        # Пересчёт вероятностей в листьях по hold-out
        self._refit_leaf_proba(self.root_, X_val, y_val)
        return self

    def _refit_leaf_proba(self, node: Node, X: pd.DataFrame, y: np.ndarray) -> None:
        """Распространяет hold-out объекты в листья и обновляет proba."""
        for i in range(len(X)):
            self._route_to_leaves(node, X.iloc[i], y[i], 1.0)
        self._apply_holdout_proba(node)

    def _route_to_leaves(
        self, node: Node, xi: pd.Series, yi: int, weight: float
    ) -> None:
        """Маршрутизация объекта в листья с накоплением весов (для hold-out)."""
        if node.label is not None:
            if not hasattr(node, "_holdout_counts"):
                node._holdout_counts = {}
            node._holdout_counts[yi] = node._holdout_counts.get(yi, 0) + weight
            return
        val = xi.get(node.feature, np.nan) if node.feature in xi.index else np.nan
        if pd.isna(val):
            self._route_to_leaves(node.left, xi, yi, weight * node.p_left)
            self._route_to_leaves(node.right, xi, yi, weight * node.p_right)
        elif val == 1:
            self._route_to_leaves(node.left, xi, yi, weight)
        else:
            self._route_to_leaves(node.right, xi, yi, weight)

    def _apply_holdout_proba(self, node: Node) -> None:
        """Заменяет proba в листе на оценки по hold-out (если есть)."""
        if node.label is not None:
            if hasattr(node, "_holdout_counts") and node._holdout_counts:
                total = sum(node._holdout_counts.values())
                if total > 0:
                    node.proba = {int(c): v / total for c, v in node._holdout_counts.items()}
                    node.label = max(node.proba, key=node.proba.get)
            return
        if node.left is not None:
            self._apply_holdout_proba(node.left)
            self._apply_holdout_proba(node.right)

    def predict_proba_single(
        self, x: pd.Series, node: Node, weight: float = 1.0
    ) -> Dict[int, float]:
        """Рекурсивный прогноз вероятностей с учётом пропусков."""
        if node.label is not None and node.proba is not None:
            return {c: p * weight for c, p in node.proba.items()}

        feat = node.feature
        val = x.get(feat, np.nan) if feat in x.index else np.nan

        if pd.isna(val):
            # Пропуск: распределяем по p_left, p_right
            proba_left = self.predict_proba_single(x, node.left, weight * node.p_left)
            proba_right = self.predict_proba_single(x, node.right, weight * node.p_right)
            merged = {}
            for c in set(proba_left) | set(proba_right):
                merged[c] = proba_left.get(c, 0) + proba_right.get(c, 0)
            return merged

        if val == 1:
            return self.predict_proba_single(x, node.left, weight)
        return self.predict_proba_single(x, node.right, weight)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        n_classes = len(self.classes_)
        probs = np.zeros((n, n_classes))
        for i in range(n):
            p = self.predict_proba_single(X.iloc[i], self.root_)
            for j, c in enumerate(self.classes_):
                probs[i, j] = p.get(c, 0)
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

    def get_depth(self, node: Optional[Node] = None) -> int:
        if node is None:
            node = self.root_
        if node is None or node.label is not None:
            return 0
        return 1 + max(
            self.get_depth(node.left),
            self.get_depth(node.right)
        )

    def count_leaves(self, node: Optional[Node] = None) -> int:
        if node is None:
            node = self.root_
        if node is None:
            return 0
        if node.label is not None:
            return 1
        return self.count_leaves(node.left) + self.count_leaves(node.right)

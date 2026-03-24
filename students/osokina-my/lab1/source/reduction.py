"""
Редукция дерева (reduced error pruning).

reduced_error_pruning(tree, X_val, y_val): post-order; для узлов с двумя листовыми детьми
пробует заменить поддерево на лист (мажоритарный класс); откат, если accuracy на X_val падает.
"""

import numpy as np
import pandas as pd
from typing import Optional
from .tree import ID3Tree, Node


def _accuracy(tree: ID3Tree, X: pd.DataFrame, y: np.ndarray) -> float:
    preds = tree.predict(X)
    return np.mean(preds == y)


def reduced_error_pruning(
    tree: ID3Tree,
    X_val: pd.DataFrame,
    y_val: np.ndarray
) -> ID3Tree:
    """
    Редукция дерева по отложенной выборке (reduced error pruning).
    Post-order: для каждого узла с двумя листовыми детьми пробуем заменить
    поддерево на лист. Мажоритарный класс — по детским листьям (samples).
    """
    def prune(node: Node) -> None:
        if node.label is not None:
            return
        if node.left is None or node.right is None:
            return
        prune(node.left)
        prune(node.right)
        if node.left.label is None or node.right.label is None:
            return
        acc_before = _accuracy(tree, X_val, y_val)
        feat_bak = node.feature
        left_bak, right_bak = node.left, node.right
        n_left = left_bak.samples
        n_right = right_bak.samples
        maj_label = left_bak.label if n_left >= n_right else right_bak.label
        node.left = None
        node.right = None
        node.feature = None
        node.label = maj_label
        node.proba = left_bak.proba if n_left >= n_right else right_bak.proba
        acc_after = _accuracy(tree, X_val, y_val)
        if acc_after < acc_before:
            node.label = None
            node.proba = None
            node.left = left_bak
            node.right = right_bak
            node.feature = feat_bak

    if tree.root_ is not None:
        prune(tree.root_)
    return tree

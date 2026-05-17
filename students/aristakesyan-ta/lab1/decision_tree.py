import numpy as np
import copy
from collections import Counter


def gini(y):
    """Индекс Джини для массива меток."""
    n = len(y)
    if n == 0:
        return 0
    counts = Counter(y)
    return 1 - sum((c / n) ** 2 for c in counts.values())


def gini_split(y_left, y_right):
    """Взвешенный Джини после разбиения на два подмножества."""
    n = len(y_left) + len(y_right)
    return (len(y_left) / n) * gini(y_left) + \
           (len(y_right) / n) * gini(y_right)


def best_split(X, y):
    """Перебор всех признаков и порогов с пропуском NaN."""
    best_gain = -1
    best_feature, best_threshold = None, None

    for feature_idx in range(X.shape[1]):
        col = X[:, feature_idx]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue

        base_gini = gini(y[valid])
        thresholds = np.unique(col[valid])

        for threshold in thresholds:
            left_mask = valid & (col <= threshold)
            right_mask = valid & (col > threshold)
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue
            gain = base_gini - gini_split(y[left_mask], y[right_mask])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


def _merge_proba(p1, p2, w1, w2):
    """Взвешенное смешивание двух словарей вероятностей."""
    all_classes = set(p1) | set(p2)
    return {c: w1 * p1.get(c, 0) + w2 * p2.get(c, 0) for c in all_classes}


class DecisionTreeClassifier:
    """Классификатор на основе дерева решений (ID3, Gini, с обработкой NaN)."""

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y[~np.isnan(y)] if y.dtype.kind == 'f' else y)
        self.tree = self._grow_tree(np.array(X, dtype=float), y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]

        counts = Counter(y)
        total = sum(counts.values())
        proba = {cls: cnt / total for cls, cnt in counts.items()}
        majority = counts.most_common(1)[0][0]

        if n_samples < self.min_samples_split or depth >= self.max_depth or len(counts) == 1:
            return {'leaf': majority, 'proba': proba}

        best_feature, best_threshold, best_gain = best_split(X, y)
        if best_feature is None or best_gain <= 0:
            return {'leaf': majority, 'proba': proba}

        col = X[:, best_feature]
        valid = ~np.isnan(col)
        left_mask = valid & (col <= best_threshold)
        right_mask = valid & (col > best_threshold)

        n_left = left_mask.sum()
        n_right = right_mask.sum()
        q_left = n_left / (n_left + n_right)
        q_right = n_right / (n_left + n_right)

        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'q_left': q_left,
            'q_right': q_right,
            'left': self._grow_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._grow_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        proba = self._get_proba(self.tree, x)
        return max(proba, key=proba.get)

    def predict_proba(self, X):
        """Возвращает массив вероятностей (n_samples, n_classes), порядок — self.classes_."""
        X = np.array(X, dtype=float)
        rows = []
        for x in X:
            p = self._get_proba(self.tree, x)
            rows.append([p.get(c, 0.0) for c in self.classes_])
        return np.array(rows)

    def _get_proba(self, node, x):
        """Рекурсивный спуск по дереву с обработкой NaN через взвешивание веток."""
        if 'leaf' in node:
            return node['proba']

        val = x[node['feature']]

        if np.isnan(val):
            p_left = self._get_proba(node['left'], x)
            p_right = self._get_proba(node['right'], x)
            return _merge_proba(p_left, p_right, node['q_left'], node['q_right'])

        if val <= node['threshold']:
            return self._get_proba(node['left'], x)
        else:
            return self._get_proba(node['right'], x)


def count_nodes(node):
    """Подсчёт количества узлов в дереве."""
    if 'leaf' in node:
        return 1
    return 1 + count_nodes(node['left']) + count_nodes(node['right'])


def count_leaves(node):
    """Подсчёт количества листьев в дереве."""
    if 'leaf' in node:
        return 1
    return count_leaves(node['left']) + count_leaves(node['right'])


def reduced_error_pruning(clf, X_val, y_val):
    """
    Reduced Error Pruning: обходим дерево снизу вверх,
    пробуем заменить каждый внутренний узел на лист.
    Если accuracy на валидации не падает — оставляем лист.
    """
    from sklearn.metrics import accuracy_score

    clf_pruned = copy.deepcopy(clf)

    def _prune(node):
        if 'leaf' in node:
            return node

        node['left'] = _prune(node['left'])
        node['right'] = _prune(node['right'])

        acc_before = accuracy_score(y_val, clf_pruned.predict(X_val))

        saved = {k: node[k] for k in list(node.keys())}

        leaf_labels = _collect_leaf_labels(node, X_val, y_val)
        if len(leaf_labels) == 0:
            return node

        leaf_counts = Counter(leaf_labels)
        leaf_total = sum(leaf_counts.values())
        leaf_majority = leaf_counts.most_common(1)[0][0]
        leaf_proba = {cls: cnt / leaf_total for cls, cnt in leaf_counts.items()}

        node.clear()
        node['leaf'] = leaf_majority
        node['proba'] = leaf_proba

        acc_after = accuracy_score(y_val, clf_pruned.predict(X_val))

        if acc_after < acc_before:
            node.clear()
            node.update(saved)

        return node

    def _collect_leaf_labels(node, X, y):
        """Собирает истинные метки объектов, проходящих через этот узел."""
        if len(X) == 0:
            return []
        if 'leaf' in node:
            return list(y)

        labels = []
        for i in range(len(X)):
            labels.append(y[i])
        return labels

    _prune(clf_pruned.tree)
    return clf_pruned

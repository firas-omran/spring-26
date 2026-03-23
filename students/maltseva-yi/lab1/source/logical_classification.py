import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeID3:

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None

    def _gini(self, y):
        if len(y) == 0:
            return 0
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p ** 2)

    def _gain(self, y, left_mask, right_mask):
        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        if n_left == 0 or n_right == 0:
            return 0
        gini_parent = self._gini(y)
        gini_left = self._gini(y[left_mask])
        gini_right = self._gini(y[right_mask])
        return gini_parent - (n_left / n) * gini_left - (n_right / n) * gini_right

    def _best_split(self, X, y, feature_names):
        best_gain = -1
        best_split = None
        for idx, name in enumerate(feature_names):
            col = X[:, idx]
            not_nan = ~pd.isna(col)
            X_sub = X[not_nan]
            y_sub = y[not_nan]
            col_sub = col[not_nan]

            if len(np.unique(col_sub)) < 2:
                continue

            if isinstance(X[0, idx], str) or (hasattr(col_sub, 'dtype') and col_sub.dtype == 'object'):
                categories = np.unique(col_sub)
                for cat in categories:
                    left_mask = (col_sub == cat)
                    right_mask = ~left_mask
                    if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                        continue
                    gain = self._gain(y_sub, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (idx, 'cat', cat)
            else:
                uniq = np.sort(np.unique(col_sub))
                thresholds = (uniq[:-1] + uniq[1:]) / 2
                for thr in thresholds:
                    left_mask = (col_sub <= thr)
                    right_mask = ~left_mask
                    if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                        continue
                    gain = self._gain(y_sub, left_mask, right_mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_split = (idx, 'num', thr)
        return best_split, best_gain

    def _build_tree(self, X, y, depth, feature_names):
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or \
           (self.max_depth is not None and depth >= self.max_depth):
            maj_class = int(np.bincount(y).argmax())
            return {'type': 'leaf', 'class': maj_class, 'n_samples': len(y)}

        split, gain = self._best_split(X, y, feature_names)
        if split is None:
            maj_class = int(np.bincount(y).argmax())
            return {'type': 'leaf', 'class': maj_class, 'n_samples': len(y)}

        idx, split_type, value = split
        feature_name = feature_names[idx]
        col = X[:, idx]

        if split_type == 'cat':
            left_mask_full = (col == value)
            right_mask_full = (col != value) & (~pd.isna(col))
        else:
            left_mask_full = (col <= value) & (~pd.isna(col))
            right_mask_full = (col > value) & (~pd.isna(col))

        n_total = len(y)
        n_left = np.sum(left_mask_full)
        n_right = np.sum(right_mask_full)

        X_left = X[left_mask_full]
        y_left = y[left_mask_full]
        X_right = X[right_mask_full]
        y_right = y[right_mask_full]

        left_child = self._build_tree(X_left, y_left, depth + 1, feature_names)
        right_child = self._build_tree(X_right, y_right, depth + 1, feature_names)

        maj_class = int(np.bincount(y).argmax())

        return {
            'type': 'internal',
            'feature': feature_name,
            'split_type': split_type,
            'value': value,
            'left': left_child,
            'right': right_child,
            'weight_left': n_left / n_total if n_total > 0 else 0,
            'weight_right': n_right / n_total if n_total > 0 else 0,
            'majority_class': maj_class,
            'n_samples': len(y)
        }

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        self.tree = self._build_tree(X, y, 0, feature_names)

    def _predict_one(self, x, node):
        if node['type'] == 'leaf':
            return node['class']

        f_name = node['feature']
        idx = self.feature_names.index(f_name)
        val = x[idx]

        if pd.isna(val):
            prob = np.zeros(2)
            left_class = self._predict_one(x, node['left'])
            right_class = self._predict_one(x, node['right'])
            prob[left_class] += node['weight_left']
            prob[right_class] += node['weight_right']
            return int(np.argmax(prob))

        if node['split_type'] == 'cat':
            if val == node['value']:
                return self._predict_one(x, node['left'])
            else:
                return self._predict_one(x, node['right'])
        else:
            if val <= node['value']:
                return self._predict_one(x, node['left'])
            else:
                return self._predict_one(x, node['right'])

    def predict(self, X):
        if self.feature_names is None:
            raise ValueError("Модель не обучена")
        preds = []
        for i in range(len(X)):
            preds.append(self._predict_one(X[i], self.tree))
        return np.array(preds)

    def _get_majority_class_in_node(self, node):
        if node['type'] == 'leaf':
            return node['class']
        return node['majority_class']

    def _predict_subtree(self, node, X_val):
        preds = []
        for i in range(len(X_val)):
            preds.append(self._predict_one_subtree(X_val[i], node))
        return np.array(preds)

    def _predict_one_subtree(self, x, node):
        if node['type'] == 'leaf':
            return node['class']
        f_name = node['feature']
        idx = self.feature_names.index(f_name)
        val = x[idx]
        if pd.isna(val):
            prob = np.zeros(2)
            left_class = self._predict_one_subtree(x, node['left'])
            right_class = self._predict_one_subtree(x, node['right'])
            prob[left_class] += node['weight_left']
            prob[right_class] += node['weight_right']
            return int(np.argmax(prob))
        if node['split_type'] == 'cat':
            if val == node['value']:
                return self._predict_one_subtree(x, node['left'])
            else:
                return self._predict_one_subtree(x, node['right'])
        else:
            if val <= node['value']:
                return self._predict_one_subtree(x, node['left'])
            else:
                return self._predict_one_subtree(x, node['right'])

    def prune(self, X_val, y_val):
        self._prune_node(self.tree, X_val, y_val)

    def _prune_node(self, node, X_val, y_val):
        if node['type'] == 'leaf':
            return

        self._prune_node(node['left'], X_val, y_val)
        self._prune_node(node['right'], X_val, y_val)

        pred_before = self._predict_subtree(node, X_val)
        err_before = np.mean(pred_before != y_val)

        leaf_class = node['majority_class']
        pred_leaf = np.full(len(X_val), leaf_class)
        err_leaf = np.mean(pred_leaf != y_val)

        if err_leaf <= err_before:
            node.clear()
            node.update({
                'type': 'leaf',
                'class': leaf_class,
                'n_samples': node.get('n_samples', 0)
            })

def load_and_preprocess_titanic():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'
    df = df[features + [target]].copy()

    df['Sex'] = df['Sex'].astype(str)
    df['Embarked'] = df['Embarked'].astype(str)

    X = df[features].values
    y = df[target].values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    feature_names = features
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def evaluate_model(model, X, y, name="Model"):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    print(f"{name:20} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    return acc, prec, rec, f1


def main():
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess_titanic()
    print(f"Размеры: train={X_train.shape[0]}, val={X_val.shape[0]}, test={X_test.shape[0]}")

    print("\nОбучение собственного ID3 (без редукции)")
    tree_id3 = DecisionTreeID3(max_depth=None, min_samples_split=2)
    tree_id3.fit(X_train, y_train, feature_names)
    print("Оценка на валидации:")
    evaluate_model(tree_id3, X_val, y_val, "ID3 (val)")

    print("\nПрименение редукции (REP)")
    tree_id3.prune(X_val, y_val)
    print("Оценка на валидации после редукции:")
    evaluate_model(tree_id3, X_val, y_val, "ID3 pruned (val)")

    print("\nОценка на тестовой выборке")
    print("ID3 без редукции:")
    evaluate_model(tree_id3, X_test, y_test, "ID3 (test)")
    print("ID3 с редукцией:")
    tree_id3_pruned = DecisionTreeID3(max_depth=None, min_samples_split=2)
    tree_id3_pruned.fit(X_train, y_train, feature_names)
    tree_id3_pruned.prune(X_val, y_val)
    evaluate_model(tree_id3_pruned, X_test, y_test, "ID3 pruned (test)")

    print("\nЭталонная реализация sklearn.tree.DecisionTreeClassifier")
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    X_train_sk = X_train.copy()
    X_val_sk = X_val.copy()
    X_test_sk = X_test.copy()

    age_median = np.nanmedian(X_train_sk[:, feature_names.index('Age')])
    for arr in [X_train_sk, X_val_sk, X_test_sk]:
        age_col = arr[:, feature_names.index('Age')]
        age_col[pd.isna(age_col)] = age_median

    embarked_mode = pd.Series(X_train_sk[:, feature_names.index('Embarked')]).mode()[0]
    for arr in [X_train_sk, X_val_sk, X_test_sk]:
        emb_col = arr[:, feature_names.index('Embarked')]
        emb_col[pd.isna(emb_col)] = embarked_mode

    df_train = pd.DataFrame(X_train_sk, columns=feature_names)
    df_val = pd.DataFrame(X_val_sk, columns=feature_names)
    df_test = pd.DataFrame(X_test_sk, columns=feature_names)

    cat_features = ['Sex', 'Embarked']
    num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='constant', fill_value=0), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ])

    X_train_enc = preprocessor.fit_transform(df_train)
    X_val_enc = preprocessor.transform(df_val)
    X_test_enc = preprocessor.transform(df_test)

    sk_tree = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=2, random_state=42)
    sk_tree.fit(X_train_enc, y_train)

    print("Оценка на валидации (sklearn):")
    y_pred_val = sk_tree.predict(X_val_enc)
    acc_val = accuracy_score(y_val, y_pred_val)
    prec_val = precision_score(y_val, y_pred_val, zero_division=0)
    rec_val = recall_score(y_val, y_pred_val, zero_division=0)
    f1_val = f1_score(y_val, y_pred_val, zero_division=0)
    print(f"sklearn (val)          | Accuracy: {acc_val:.4f} | Precision: {prec_val:.4f} | Recall: {rec_val:.4f} | F1: {f1_val:.4f}")

    print("Оценка на тесте (sklearn):")
    y_pred_test = sk_tree.predict(X_test_enc)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    print(f"sklearn (test)         | Accuracy: {acc_test:.4f} | Precision: {prec_test:.4f} | Recall: {rec_test:.4f} | F1: {f1_test:.4f}")

    print("\nСводная таблица результатов (тестовая выборка)")
    print("Модель                 | Accuracy | Precision | Recall | F1")
    tree_no_prune = DecisionTreeID3(max_depth=None, min_samples_split=2)
    tree_no_prune.fit(X_train, y_train, feature_names)
    acc_np, prec_np, rec_np, f1_np = evaluate_model(tree_no_prune, X_test, y_test, "ID3 (no prune)")
    acc_pr, prec_pr, rec_pr, f1_pr = evaluate_model(tree_id3_pruned, X_test, y_test, "ID3 (pruned)")
    print(f"sklearn                | {acc_test:.4f}    | {prec_test:.4f}     | {rec_test:.4f}   | {f1_test:.4f}")

    models = ['ID3 (no prune)', 'ID3 (pruned)', 'sklearn']
    acc_vals = [acc_np, acc_pr, acc_test]
    prec_vals = [prec_np, prec_pr, prec_test]
    rec_vals = [rec_np, rec_pr, rec_test]
    f1_vals = [f1_np, f1_pr, f1_test]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - 1.5*width, acc_vals, width, label='Accuracy')
    rects2 = ax.bar(x - 0.5*width, prec_vals, width, label='Precision')
    rects3 = ax.bar(x + 0.5*width, rec_vals, width, label='Recall')
    rects4 = ax.bar(x + 1.5*width, f1_vals, width, label='F1')

    ax.set_ylabel('Score')
    ax.set_title('Сравнение метрик качества на тестовой выборке')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    plt.tight_layout()
    fig.savefig('images/metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


    y_pred_no_prune = tree_no_prune.predict(X_test)
    y_pred_pruned = tree_id3_pruned.predict(X_test)

    cm_no_prune = confusion_matrix(y_test, y_pred_no_prune)
    cm_pruned = confusion_matrix(y_test, y_pred_pruned)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    titles = ['До pruning', 'После pruning']
    cms = [cm_no_prune, cm_pruned]
    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig('images/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
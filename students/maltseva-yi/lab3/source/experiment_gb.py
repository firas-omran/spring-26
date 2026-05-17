import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.tree import DecisionTreeRegressor

from gb_custom import GradientBoostingCustom
from adult_data import load_adult_data, get_preprocessor

def plot_loss_curve(train_loss, test_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_loss)+1), train_loss, label='Train Log Loss')
    plt.plot(range(1, len(test_loss)+1), test_loss, label='Test Log Loss')
    plt.xlabel('Number of trees')
    plt.ylabel('Log Loss')
    plt.title('Gradient Boosting: Loss vs. Iterations')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/gb_loss_curve.png', dpi=150)
    plt.show()

def shorten_names(names, max_len=40):
    # Обрезает длинные имена признаков для читаемости.
    return [name if len(name) <= max_len else name[:max_len-3] + '...' for name in names]

def main():
    X, y = load_adult_data()
    print(f"Размер полного датасета: {X.shape}")
    print(f"Доля >50K: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor, num_feats, cat_feats = get_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train).toarray()
    X_test_prep = preprocessor.transform(X_test).toarray()
    print(f"Train после кодирования: {X_train_prep.shape}, Test: {X_test_prep.shape}")

    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'random_state': 42
    }

    # Собственная модель 
    gb_custom = GradientBoostingCustom(**params)
    start = time.time()
    gb_custom.fit(X_train_prep, y_train)
    custom_time = time.time() - start
    y_pred_custom = gb_custom.predict(X_test_prep)
    proba_custom = gb_custom.predict_proba(X_test_prep)
    custom_acc = accuracy_score(y_test, y_pred_custom)
    custom_auc = roc_auc_score(y_test, proba_custom[:, 1])
    custom_logloss = log_loss(y_test, proba_custom)
    print(f"\nCustom GB | Acc: {custom_acc:.4f}, AUC: {custom_auc:.4f}, LogLoss: {custom_logloss:.4f}, Time: {custom_time:.2f}s")

    # Кросс-валидация custom
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores_custom, cv_times = [], []
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_prep, y_train), 1):
        X_tr, X_val = X_train_prep[tr_idx], X_train_prep[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        m = GradientBoostingCustom(**params)
        t0 = time.time()
        m.fit(X_tr, y_tr)
        t1 = time.time()
        acc = accuracy_score(y_val, m.predict(X_val))
        cv_scores_custom.append(acc)
        cv_times.append(t1 - t0)
        print(f"  Fold {fold}: acc={acc:.4f}, time={t1-t0:.2f}s")
    print(f"Custom CV accuracy: {np.mean(cv_scores_custom):.4f} +/- {np.std(cv_scores_custom):.4f}")

    # Эталонная sklearn 
    gb_sk = GradientBoostingClassifier(**params)
    start = time.time()
    gb_sk.fit(X_train_prep, y_train)
    sk_time = time.time() - start
    y_pred_sk = gb_sk.predict(X_test_prep)
    proba_sk = gb_sk.predict_proba(X_test_prep)
    sk_acc = accuracy_score(y_test, y_pred_sk)
    sk_auc = roc_auc_score(y_test, proba_sk[:, 1])
    sk_logloss = log_loss(y_test, proba_sk)
    print(f"\nsklearn GB | Acc: {sk_acc:.4f}, AUC: {sk_auc:.4f}, LogLoss: {sk_logloss:.4f}, Time: {sk_time:.2f}s")

    cv_scores_sk = cross_val_score(gb_sk, X_train_prep, y_train, cv=cv, scoring='accuracy')
    print(f"sklearn CV accuracy: {np.mean(cv_scores_sk):.4f} +/- {np.std(cv_scores_sk):.4f}")

    print("\n===== Итоговое сравнение (тестовая выборка) =====")
    print(f"{'Модель':<25} {'Acc':>8} {'AUC':>8} {'LogLoss':>8} {'Время':>8}")
    print(f"{'Custom GB':<25} {custom_acc:>8.4f} {custom_auc:>8.4f} {custom_logloss:>8.4f} {custom_time:>8.2f}")
    print(f"{'sklearn GB':<25} {sk_acc:>8.4f} {sk_auc:>8.4f} {sk_logloss:>8.4f} {sk_time:>8.2f}")

    # Кривая log loss по итерациям
    X_sub, X_val, y_sub, y_val = train_test_split(X_train_prep, y_train, test_size=0.2, random_state=42)
    gb_curve = GradientBoostingCustom(**params)
    gb_curve.init_pred = np.log(np.clip(np.mean(y_sub), 1e-15, 1-1e-15))
    f_sub = np.full(X_sub.shape[0], gb_curve.init_pred)
    f_val = np.full(X_val.shape[0], gb_curve.init_pred)
    train_loss, val_loss = [], []
    rng = np.random.RandomState(params['random_state'])
    for t in range(params['n_estimators']):
        proba_sub = gb_curve._sigmoid(f_sub)
        residuals = gb_curve._log_loss_gradient(y_sub, proba_sub)
        if gb_curve.subsample < 1.0:
            idx = rng.choice(len(X_sub), int(len(X_sub)*gb_curve.subsample), replace=False)
            X_ss, r_ss = X_sub[idx], residuals[idx]
        else:
            X_ss, r_ss = X_sub, residuals
        tree = DecisionTreeRegressor(max_depth=gb_curve.max_depth, random_state=rng.randint(0,1e6))
        tree.fit(X_ss, r_ss)
        f_sub += gb_curve.learning_rate * tree.predict(X_sub)
        f_val += gb_curve.learning_rate * tree.predict(X_val)
        train_loss.append(log_loss(y_sub, gb_curve._sigmoid(f_sub)))
        val_loss.append(log_loss(y_val, gb_curve._sigmoid(f_val)))
    plot_loss_curve(train_loss, val_loss)

    # Важность признаков (топ-20)
    feature_names = preprocessor.get_feature_names_out()
    n = min(len(feature_names), len(gb_custom.feature_importances_), len(gb_sk.feature_importances_))
    importances_custom = gb_custom.feature_importances_[:n]
    importances_sk = gb_sk.feature_importances_[:n]

    sorted_idx = np.argsort(importances_sk)

    top_n = 20
    top_idx = sorted_idx[-top_n:]

    top_feature_names = shorten_names(np.array(feature_names)[top_idx], max_len=45)

    plt.figure(figsize=(10, 6))
    plt.barh(top_feature_names, importances_sk[top_idx], color='skyblue', label='sklearn')
    plt.barh(top_feature_names, importances_custom[top_idx], color='orange', alpha=0.6, label='custom')
    plt.xlabel('Feature Importance')
    plt.title(f'Топ-{top_n} важнейших признаков')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/gb_feature_importance_top.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
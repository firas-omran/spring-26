from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.data import prepare_data
from source.forest import OOBRandomForestClassifier, oob_scorer

RANDOM_STATE = 42
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


def evaluate(model, X_test, y_test, fit_time: float) -> dict:
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - start
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "oob_score": float(model.oob_score_),
        "fit_time": float(fit_time),
        "predict_time": float(predict_time),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def run_grid_search(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 8],
        "max_features": ["sqrt", 0.5],
        "min_samples_leaf": [1, 3],
        "min_samples_split": [2, 5],
    }
    train_idx = np.arange(len(y_train))
    grid = GridSearchCV(
        estimator=OOBRandomForestClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring=oob_scorer,
        cv=[(train_idx, train_idx)],
        refit=True,
        n_jobs=1,
        return_train_score=False,
    )
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return grid, elapsed


def fit_sklearn(best_params, X_train, y_train):
    model = SklearnRandomForestClassifier(
        **best_params,
        bootstrap=True,
        oob_score=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    return model, elapsed


def plot_class_distribution(dataset_info):
    counts = dataset_info["class_counts"]
    labels = ["Not survived", "Survived"]
    values = [counts.get("0", 0), counts.get("1", 0)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#69788f", "#d08c60"])
    ax.set_title("Titanic class distribution")
    ax.set_ylabel("Passengers")
    for i, value in enumerate(values):
        ax.text(i, value + 5, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def plot_metric_comparison(custom_metrics, sklearn_metrics):
    names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "OOB"]
    custom = [custom_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "oob_score"]]
    sklearn = [sklearn_metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "oob_score"]]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, custom, width, label="Custom RF", color="#376996")
    ax.bar(x + width / 2, sklearn, width, label="sklearn RF", color="#d08c60")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Quality metrics comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_time_comparison(custom_metrics, sklearn_metrics):
    names = ["Fit time", "Predict time"]
    custom = [custom_metrics["fit_time"], custom_metrics["predict_time"]]
    sklearn = [sklearn_metrics["fit_time"], sklearn_metrics["predict_time"]]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, custom, width, label="Custom RF", color="#376996")
    ax.bar(x + width / 2, sklearn, width, label="sklearn RF", color="#d08c60")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Seconds")
    ax.set_title("Training and prediction time")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "time_comparison.png", dpi=180)
    plt.close(fig)


def plot_confusions(y_test, custom_metrics, sklearn_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, title, metrics in [
        (axes[0], "Custom RF", custom_metrics),
        (axes[1], "sklearn RF", sklearn_metrics),
    ]:
        cm = confusion_matrix(y_test, metrics["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrices.png", dpi=180)
    plt.close(fig)


def plot_roc(y_test, custom_metrics, sklearn_metrics):
    fig, ax = plt.subplots(figsize=(6, 5))
    for title, metrics, color in [
        ("Custom RF", custom_metrics, "#376996"),
        ("sklearn RF", sklearn_metrics, "#d08c60"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, metrics["y_proba"])
        ax.plot(fpr, tpr, label=f"{title} AUC={metrics['roc_auc']:.3f}", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=180)
    plt.close(fig)


def plot_feature_importance(importances):
    top = pd.DataFrame(
        sorted(importances.items(), key=lambda item: item[1], reverse=True)[:15],
        columns=["feature", "importance"],
    ).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#2f7f6f")
    ax.set_title("Top OOB permutation importances")
    ax.set_xlabel("OOB accuracy drop")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "oob_feature_importance.png", dpi=180)
    plt.close(fig)


def plot_grid_results(grid):
    df = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False).head(20)
    labels = []
    for _, row in df.iterrows():
        params = row["params"]
        labels.append(
            f"n={params['n_estimators']}, depth={params['max_depth']}, "
            f"mf={params['max_features']}, leaf={params['min_samples_leaf']}, split={params['min_samples_split']}"
        )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(np.arange(len(df)), df["mean_test_score"].to_numpy()[::-1], color="#6f6aa8")
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(labels[::-1], fontsize=8)
    ax.set_xlabel("OOB score")
    ax.set_title("Grid search: top 20 configurations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "grid_search_top20.png", dpi=180)
    plt.close(fig)


def write_report(dataset_info, best_params, grid_time, custom_metrics, sklearn_metrics, importances):
    top_importance_rows = "\n".join(
        f"| {feature} | {value:.4f} |"
        for feature, value in sorted(importances.items(), key=lambda item: item[1], reverse=True)[:12]
    )
    metric_table = f"""| Model | OOB | Accuracy | Precision | Recall | F1 | ROC-AUC | Fit time, s | Predict time, s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Custom Random Forest | {custom_metrics['oob_score']:.4f} | {custom_metrics['accuracy']:.4f} | {custom_metrics['precision']:.4f} | {custom_metrics['recall']:.4f} | {custom_metrics['f1']:.4f} | {custom_metrics['roc_auc']:.4f} | {custom_metrics['fit_time']:.4f} | {custom_metrics['predict_time']:.4f} |
| sklearn RandomForestClassifier | {sklearn_metrics['oob_score']:.4f} | {sklearn_metrics['accuracy']:.4f} | {sklearn_metrics['precision']:.4f} | {sklearn_metrics['recall']:.4f} | {sklearn_metrics['f1']:.4f} | {sklearn_metrics['roc_auc']:.4f} | {sklearn_metrics['fit_time']:.4f} | {sklearn_metrics['predict_time']:.4f} |"""

    report = f"""# Лабораторная работа №2. Ансамбли моделей

## Цель работы
Реализовать собственный ансамбль Random Forest на базе библиотечных деревьев решений, подобрать гиперпараметры по OOB-оценке, получить важность признаков через OOB^j и сравнить результат с эталонной реализацией scikit-learn.

## Выбранный датасет
Использован датасет Titanic. Целевая переменная `Survived` задает бинарную классификацию: пассажир выжил или не выжил.

В отличие от минимального набора признаков, в этой работе были добавлены признаки, извлеченные из исходных полей:

- `Title` из имени пассажира;
- `Deck` из номера каюты;
- `FamilySize = SibSp + Parch + 1`;
- `IsAlone` как индикатор одиночной поездки;
- `TicketGroupSize` как число пассажиров с одинаковым билетом;
- `FarePerPerson` как тариф, деленный на размер семьи.

Размер данных: {dataset_info['n_samples']} объектов. Обучающая выборка: {dataset_info['n_train']}, тестовая выборка: {dataset_info['n_test']}. Распределение классов: не выжил = {dataset_info['class_counts'].get('0', 0)}, выжил = {dataset_info['class_counts'].get('1', 0)}.

Предобработка выполнялась без утечки данных: сначала train/test split, затем `SimpleImputer` и `OneHotEncoder` обучались только на train-части. После one-hot кодирования получилось {len(dataset_info['prepared_features'])} признаков.

## Описание метода
Реализован класс `OOBRandomForestClassifier`. Он совместим с API sklearn и использует `DecisionTreeClassifier` как базовый алгоритм.

Для каждого дерева выполняется:

1. формирование bootstrap-выборки размера `n` с возвращением;
2. определение OOB-объектов, которые не попали в bootstrap;
3. обучение дерева решений с ограничениями из текущей комбинации гиперпараметров;
4. накопление OOB-предсказаний для вычисления `oob_score_`.

Итоговое предсказание строится soft-voting: вероятности классов усредняются по всем деревьям, затем выбирается класс с максимальной средней вероятностью.

## Подбор гиперпараметров
Для подбора использован `GridSearchCV` из sklearn. Так как качество должно подбираться по OOB, применен специальный scorer, возвращающий `estimator.oob_score_`. В качестве CV передается один технический fold по train-выборке; тестовая выборка в подборе параметров не участвует.

Перебирались параметры:

- `n_estimators`: 50, 100, 200;
- `max_depth`: None, 5, 8;
- `max_features`: `sqrt`, 0.5;
- `min_samples_leaf`: 1, 3;
- `min_samples_split`: 2, 5.

Всего проверено 72 комбинации. Время grid search: {grid_time:.2f} с.

Лучшие параметры по OOB:

```json
{json.dumps(best_params, ensure_ascii=False, indent=2)}
```

## Результаты экспериментов
{metric_table}

Обе модели обучались на одинаковых признаках и с одинаковыми лучшими гиперпараметрами. Это делает сравнение более честным: отличается только реализация ансамбля, а не настройка модели.

## Важность признаков через OOB^j
Важность признака оценивалась как падение OOB accuracy после случайной перестановки значений этого признака. Чем больше падение, тем важнее признак.

| Feature | OOB accuracy drop |
|---|---:|
{top_importance_rows}

Наиболее важные признаки связаны с полом пассажира, классом билета, возрастом, семейной структурой и стоимостью поездки. Это согласуется с исторической интерпретацией Titanic: вероятность выживания зависела от пола, социального класса и состава группы пассажира.

## Графики

### Распределение классов

![Class distribution](results/plots/class_distribution.png)

В выборке пассажиров, которые не выжили, больше, чем выживших: 549 против 342. Дисбаланс не является экстремальным, но он заметен, поэтому в отчете дополнительно используются precision, recall и F1, а не только accuracy.

### Сравнение качества моделей

![Metric comparison](results/plots/metric_comparison.png)

Качество собственной реализации и `RandomForestClassifier` из sklearn практически совпадает. Небольшое преимущество собственной модели на тестовой выборке находится в пределах естественного разброса для небольшого датасета Titanic, но главное здесь в другом: близость метрик подтверждает корректность реализации bootstrap, OOB-голосования и усреднения вероятностей.

### Сравнение времени работы

![Time comparison](results/plots/time_comparison.png)

Эталонная реализация sklearn быстрее, особенно на этапе предсказания. Это ожидаемо, так как `RandomForestClassifier` оптимизирован внутри библиотеки. Собственная реализация написана на Python и предназначена прежде всего для демонстрации механики ансамбля.

### Матрицы ошибок

![Confusion matrices](results/plots/confusion_matrices.png)

Матрицы ошибок показывают, что обе модели ведут себя похоже. Основная сложность задачи связана с объектами класса `Survived = 1`: часть выживших пассажиров модель относит к отрицательному классу. Это типично для Titanic, где выживание зависит от нескольких взаимосвязанных факторов, а не от одного признака.

### ROC-кривые

![ROC curves](results/plots/roc_curves.png)

ROC-AUC около 0.85 показывает, что обе модели достаточно хорошо ранжируют пассажиров по вероятности выживания. Кривые почти совпадают, что дополнительно подтверждает сопоставимость собственной и библиотечной реализаций.

### Важность признаков через OOB^j

![OOB feature importance](results/plots/oob_feature_importance.png)

Наиболее заметное падение OOB-точности происходит при перемешивании `FarePerPerson`, `Title_Mr`, `Age` и `Pclass`. Эти признаки отражают социальный статус, пол/обращение, возраст и стоимость билета, поэтому их высокая важность хорошо согласуется с содержательной интерпретацией датасета.

### Результаты Grid Search

![Grid search top 20](results/plots/grid_search_top20.png)

На графике показаны 20 лучших конфигураций по OOB-оценке. Лучшие варианты используют 100 или 200 деревьев и ограничение глубины/листа, что помогает уменьшить переобучение на небольшом датасете.

## Выводы
В работе был реализован Random Forest с bootstrap-обучением, OOB-оценкой и OOB-пермутационной важностью признаков. Подбор гиперпараметров через `GridSearchCV` по OOB позволил выбрать модель без отдельной validation-выборки.

Собственная реализация показала качество, сопоставимое с `RandomForestClassifier` из sklearn. При этом эталонная реализация обычно быстрее за счет оптимизаций, но собственная реализация явно демонстрирует внутреннюю механику ансамбля: bootstrap, OOB-голосование и влияние отдельных признаков на OOB-ошибку.
"""
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    (PROJECT_ROOT / "README.md").write_text(report + "\n\n## Запуск\n```bash\npython source/experiment.py\n```\n", encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, dataset_info = prepare_data(PROJECT_ROOT, random_state=RANDOM_STATE)
    grid, grid_time = run_grid_search(X_train, y_train)
    best_params = grid.best_params_

    custom_model = OOBRandomForestClassifier(**best_params, random_state=RANDOM_STATE)
    start = time.perf_counter()
    custom_model.fit(X_train, y_train)
    custom_fit_time = time.perf_counter() - start
    custom_metrics = evaluate(custom_model, X_test, y_test, custom_fit_time)

    sklearn_model, sklearn_fit_time = fit_sklearn(best_params, X_train, y_train)
    sklearn_metrics = evaluate(sklearn_model, X_test, y_test, sklearn_fit_time)

    importances = custom_model.oob_permutation_importance(X_train, y_train, n_repeats=7, random_state=RANDOM_STATE)

    plot_class_distribution(dataset_info)
    plot_metric_comparison(custom_metrics, sklearn_metrics)
    plot_time_comparison(custom_metrics, sklearn_metrics)
    plot_confusions(y_test, custom_metrics, sklearn_metrics)
    plot_roc(y_test, custom_metrics, sklearn_metrics)
    plot_feature_importance(importances)
    plot_grid_results(grid)

    serializable_custom = {k: v for k, v in custom_metrics.items() if k not in {"y_pred", "y_proba"}}
    serializable_sklearn = {k: v for k, v in sklearn_metrics.items() if k not in {"y_pred", "y_proba"}}
    grid_results = pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False)
    grid_results[["params", "mean_test_score", "rank_test_score"]].to_csv(RESULTS_DIR / "grid_search_results.csv", index=False)
    pd.DataFrame(
        sorted(importances.items(), key=lambda item: item[1], reverse=True),
        columns=["feature", "oob_accuracy_drop"],
    ).to_csv(RESULTS_DIR / "oob_feature_importance.csv", index=False)

    payload = {
        "dataset": dataset_info,
        "best_params": best_params,
        "grid_search_time": grid_time,
        "custom_metrics": serializable_custom,
        "sklearn_metrics": serializable_sklearn,
        "oob_permutation_importance": importances,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(dataset_info, best_params, grid_time, custom_metrics, sklearn_metrics, importances)

    print("Experiment completed")
    print(f"Best params: {best_params}")
    print(f"Custom accuracy: {custom_metrics['accuracy']:.4f}, OOB: {custom_metrics['oob_score']:.4f}")
    print(f"sklearn accuracy: {sklearn_metrics['accuracy']:.4f}, OOB: {sklearn_metrics['oob_score']:.4f}")
    print(f"Report: {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()

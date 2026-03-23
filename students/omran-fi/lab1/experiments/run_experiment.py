from pathlib import Path
from time import perf_counter
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from sklearn.tree import DecisionTreeClassifier

from source.data_loader import load_dataset
from source.decision_tree import DecisionTree
from source.metrics import classification_report
from source.preprocessing import prepare_data
from source.visualization import plot_accuracy
from source.visualization import plot_class_distribution
from source.visualization import plot_confusion
from source.visualization import plot_depth_analysis
from source.visualization import plot_feature_importance
from source.visualization import plot_metric_comparison
from source.visualization import plot_roc_curve
from source.visualization import plot_time_comparison


def evaluate_model(y_true, y_pred, y_prob, fit_time, predict_time):

    report = classification_report(y_true, y_pred, y_prob)
    report["fit_time"] = fit_time
    report["predict_time"] = predict_time
    return report


def format_report_block(title, report):

    lines = [f"{title}\n"]

    ordered_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "fit_time",
        "predict_time",
    ]

    for key in ordered_keys:
        if key in report:
            lines.append(f"{key}: {report[key]:.4f}")

    return "\n".join(lines) + "\n"


df = load_dataset()

X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(df)


tree = DecisionTree(max_depth=10)

fit_start = perf_counter()
tree.fit(X_train, y_train)
custom_fit_time = perf_counter() - fit_start

predict_start = perf_counter()
pred_before = tree.predict(X_test)
proba_before = tree.predict_proba(X_test)
custom_predict_time_before = perf_counter() - predict_start

report_before = evaluate_model(
    y_test,
    pred_before,
    proba_before,
    custom_fit_time,
    custom_predict_time_before,
)

print("\n===== CUSTOM TREE BEFORE PRUNING =====")
for key, value in report_before.items():
    print(f"{key}: {value:.4f}")

plot_confusion(y_test, pred_before, "custom_before_pruning")


tree.prune(X_val, y_val)

predict_start = perf_counter()
pred_after = tree.predict(X_test)
proba_after = tree.predict_proba(X_test)
custom_predict_time_after = perf_counter() - predict_start

report_after = evaluate_model(
    y_test,
    pred_after,
    proba_after,
    custom_fit_time,
    custom_predict_time_after,
)

print("\n===== CUSTOM TREE AFTER PRUNING =====")
for key, value in report_after.items():
    print(f"{key}: {value:.4f}")

plot_confusion(y_test, pred_after, "custom_after_pruning")


sklearn_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=10,
    random_state=42,
)

fit_start = perf_counter()
sklearn_tree.fit(X_train.fillna(X_train.median()), y_train)
sklearn_fit_time = perf_counter() - fit_start

predict_start = perf_counter()
pred_sklearn = sklearn_tree.predict(X_test.fillna(X_train.median()))
proba_sklearn = sklearn_tree.predict_proba(X_test.fillna(X_train.median()))[:, 1]
sklearn_predict_time = perf_counter() - predict_start

report_sklearn = evaluate_model(
    y_test,
    pred_sklearn,
    proba_sklearn,
    sklearn_fit_time,
    sklearn_predict_time,
)

print("\n===== SKLEARN TREE =====")
for key, value in report_sklearn.items():
    print(f"{key}: {value:.4f}")

plot_confusion(y_test, pred_sklearn, "sklearn")


plot_accuracy(
    ("Custom Before Pruning", report_before["accuracy"]),
    ("Custom After Pruning", report_after["accuracy"]),
    ("Sklearn", report_sklearn["accuracy"]),
)

plot_metric_comparison({
    "Custom Before": {
        "accuracy": report_before["accuracy"],
        "precision": report_before["precision"],
        "recall": report_before["recall"],
        "f1": report_before["f1"],
        "auc": report_before["auc"],
    },
    "Custom After": {
        "accuracy": report_after["accuracy"],
        "precision": report_after["precision"],
        "recall": report_after["recall"],
        "f1": report_after["f1"],
        "auc": report_after["auc"],
    },
    "Sklearn": {
        "accuracy": report_sklearn["accuracy"],
        "precision": report_sklearn["precision"],
        "recall": report_sklearn["recall"],
        "f1": report_sklearn["f1"],
        "auc": report_sklearn["auc"],
    },
})

plot_time_comparison({
    "Custom Before": {
        "fit_time": report_before["fit_time"],
        "predict_time": report_before["predict_time"],
    },
    "Custom After": {
        "fit_time": report_after["fit_time"],
        "predict_time": report_after["predict_time"],
    },
    "Sklearn": {
        "fit_time": report_sklearn["fit_time"],
        "predict_time": report_sklearn["predict_time"],
    },
})

plot_depth_analysis(X_train, y_train, X_test, y_test)
plot_class_distribution(y_train)
plot_roc_curve(y_test, proba_after)
plot_feature_importance(tree.feature_importances_, X_train.columns)


with open("results/metrics.txt", "w") as file:
    file.write("===== MODEL RESULTS =====\n\n")
    file.write(format_report_block("Custom Tree Before Pruning", report_before))
    file.write("\n")
    file.write(format_report_block("Custom Tree After Pruning", report_after))
    file.write("\n")
    file.write(format_report_block("Sklearn Tree", report_sklearn))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def plot_confusion(y_true, y_pred, name):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}_confusion.png")
    plt.close()


def plot_accuracy(*model_pairs):

    labels = [label for label, _ in model_pairs]
    accuracies = [value for _, value in model_pairs]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=labels, y=accuracies, palette="deep")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("results/plots/model_comparison.png")
    plt.close()


def plot_depth_analysis(X_train, y_train, X_test, y_test):

    from .decision_tree import DecisionTree

    depths = range(1, 15)
    accuracies = []

    for depth in depths:
        tree = DecisionTree(max_depth=depth)
        tree.fit(X_train, y_train)
        predictions = tree.predict(X_test)
        accuracies.append(np.mean(predictions == y_test))

    plt.figure(figsize=(7, 4))
    plt.plot(depths, accuracies, marker="o")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.title("Depth Analysis")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/tree_depth_analysis.png")
    plt.close()


def plot_class_distribution(y, save_path="results/plots/class_distribution.png"):

    plt.figure(figsize=(6, 5))

    values = pd.Series(y).value_counts().sort_index()

    sns.barplot(
        x=["Died", "Survived"],
        y=values.values,
        palette="deep",
    )

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path="results/plots/roc_curve.png"):

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(
    importances,
    feature_names,
    save_path="results/plots/feature_importance.png",
):

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="importance", y="feature", palette="deep")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_metric_comparison(metrics_by_model, save_path="results/plots/metric_comparison.png"):

    frame = pd.DataFrame(metrics_by_model).T.reset_index().rename(columns={"index": "model"})
    metric_columns = [column for column in frame.columns if column != "model"]
    melted = frame.melt(id_vars="model", value_vars=metric_columns, var_name="metric", value_name="value")

    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="metric", y="value", hue="model", palette="deep")
    plt.ylim(0, 1)
    plt.title("Metric Comparison")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_time_comparison(times_by_model, save_path="results/plots/time_comparison.png"):

    frame = pd.DataFrame(times_by_model).T.reset_index().rename(columns={"index": "model"})
    melted = frame.melt(id_vars="model", var_name="stage", value_name="seconds")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=melted, x="stage", y="seconds", hue="model", palette="deep")
    plt.title("Training And Prediction Time")
    plt.ylabel("Seconds")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

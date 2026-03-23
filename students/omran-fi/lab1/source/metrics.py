import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def evaluate(y_true, y_pred):

    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return float(tp / (tp + fp + 1e-9))


def recall(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return float(tp / (tp + fn + 1e-9))


def f1_score(y_true, y_pred):

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return float(2 * p * r / (p + r + 1e-9))


def confusion(y_true, y_pred):

    return confusion_matrix(y_true, y_pred)


def roc_auc(y_true, y_prob):

    return float(roc_auc_score(y_true, y_prob))


def classification_report(y_true, y_pred, y_prob=None):

    report = {
        "accuracy": evaluate(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    if y_prob is not None:
        report["auc"] = roc_auc(y_true, y_prob)

    return report

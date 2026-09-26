import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, roc_curve


def binary_metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "auroc": roc_auc_score(y, p),
        "bal_acc": balanced_accuracy_score(y, p > 0.5),
        "sens@90spec": float(np.interp(0.10, fpr, tpr)),
    }


def multiclass_metrics(y, P):
    return {
        "bal_acc": balanced_accuracy_score(y, P.argmax(1)),
        "macro_f1": f1_score(y, P.argmax(1), average="macro"),
        "macro_auroc": roc_auc_score(y, P, multi_class="ovr", average="macro"),
    }


def evaluate(task, y, out):
    return binary_metrics(y, out) if task == "bin" else multiclass_metrics(y, out)

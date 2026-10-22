"""Evaluation helpers."""
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def report(name, y, pred, score, mask=None, verbose=True):
    """Metrics for one model; ``mask`` restricts scoring to a subset of rows."""
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score)
    if mask is not None:
        y, pred, score = y[mask], pred[mask], score[mask]
    if verbose:
        print(f"\n=== {name} ===")
        print(classification_report(y, pred, target_names=["benign", "malicious"], digits=4))
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, pred))
    return {
        "name": name,
        "F1": float(f1_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "AUC": float(roc_auc_score(y, score)),
        "n": int(len(y)),
    }


def best_threshold(y, p):
    """Threshold that maximises F1, and the F1 it reaches.

    Pick this on a validation split, not on the test set you report on.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    i = int(np.argmax(f1s[:-1]))
    return float(thr[i]), float(f1s[i])


def to_markdown(rows):
    """Results as a markdown table, best F1 first."""
    rows = sorted(rows, key=lambda r: r["F1"], reverse=True)
    lines = ["| Model | F1 | Precision | Recall | AUC | n |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['F1']:.4f} | {r['precision']:.4f} | "
                     f"{r['recall']:.4f} | {r['AUC']:.4f} | {r['n']} |")
    return "\n".join(lines)

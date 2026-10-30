"""Held-out test-set performance of the deployed CNN-LSTM model.

The confusion matrix is transcribed from results/confusion_matrix_cnn_lstm.png
(MIT-BIH heartbeat test split, 21,892 beats). Rows = true class, columns = predicted.
"""
import numpy as np

CONFUSION_MATRIX = [
    [18050, 39, 16, 10, 3],
    [65, 482, 8, 1, 0],
    [26, 4, 1398, 20, 0],
    [6, 0, 13, 142, 1],
    [7, 0, 0, 0, 1601],
]

FIGURES = [
    {"file": "training_history.png", "title": "Training loss & validation accuracy"},
    {"file": "confusion_matrix_cnn_lstm.png", "title": "CNN-LSTM confusion matrix (deployed model)"},
    {"file": "confusion_matrix_resnet_se.png", "title": "ResNet-SE confusion matrix"},
    {"file": "confusion_matrix_ensemble.png", "title": "Ensemble confusion matrix"},
]


def metrics_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    # Specificity: correctly rejected beats among all beats not of this class
    tn = cm.sum() - support - predicted + tp
    specificity = tn / (cm.sum() - support)
    present = support > 0
    return {
        "total": int(cm.sum()),
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1[present].mean()),
        "weighted_f1": float((f1 * support).sum() / support.sum()),
        "per_class": [
            {
                "id": i,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "specificity": float(specificity[i]),
                "support": int(support[i]),
            }
            for i in range(len(cm))
        ],
        "confusion_matrix": cm.astype(int).tolist(),
    }

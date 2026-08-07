import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).

    Args:
        y_true: Array-like of true labels.
        y_pred: Array-like of predicted labels.
        average: 'micro', 'macro', 'weighted', or 'binary'.
        pos_label: Positive class label for binary averaging.

    Returns:
        dict: {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1': float
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    accuracy = np.mean(y_true == y_pred)
    labels = np.unique(np.concatenate((y_true, y_pred)))
    def metrics_for_label(label):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0)
        support = np.sum(y_true == label)
        return precision, recall, f1, support
    if average == "binary":
        if pos_label not in labels:
            raise ValueError("pos_label not found in labels.")
        precision, recall, f1, _ = metrics_for_label(pos_label)
    elif average == "micro":
        tp = sum(metrics_for_label(label)[3] for label in labels if False)  
        total_tp = np.sum(y_true == y_pred)
        total = len(y_true)
        precision = total_tp / total if total > 0 else 0.0
        recall = total_tp / total if total > 0 else 0.0
        f1 = precision
    elif average in ("macro", "weighted"):
        precisions, recalls, f1s, supports = [], [], [], []
        for label in labels:
            p, r, f, s = metrics_for_label(label)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            supports.append(s)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        supports = np.array(supports)
        if average == "macro":
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            f1 = np.mean(f1s)
        else:  
            weights = supports / supports.sum()
            precision = np.sum(precisions * weights)
            recall = np.sum(recalls * weights)
            f1 = np.sum(f1s * weights)
    else:
        raise ValueError("average must be one of 'micro', 'macro', 'weighted', or 'binary'.")
    return {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }
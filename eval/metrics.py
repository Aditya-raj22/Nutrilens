"""Classification metrics: top-1, top-5, macro-F1, confusion matrix."""
import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, top5 = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
        top5.extend(logits.topk(5, dim=1).indices.cpu().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    top1_acc = float((y_true == y_pred).mean())
    top5_acc = float(np.mean([t in row for t, row in zip(y_true, top5)]))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "macro_f1": macro_f1,
        "confusion": confusion_matrix(y_true, y_pred),
    }


def per_class_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)

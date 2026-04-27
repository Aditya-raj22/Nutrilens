"""Full eval run on the Food-101 test split.

Writes metrics JSON and confusion matrix PNG to eval/outputs/.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from classifier.data import build_loaders
from classifier.infer import load_classifier
from eval.metrics import evaluate, per_class_f1


def plot_confusion(cm, classes, out_path, top_k=20):
    """Plot a confusion matrix restricted to the top_k most-confused class pairs."""
    cm = cm.copy()
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                pairs.append((cm[i, j], i, j))
    pairs.sort(reverse=True)
    top = pairs[:top_k]
    labels = [f"{classes[i]}->{classes[j]}" for _, i, j in top]
    counts = [c for c, _, _ in top]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels[::-1], counts[::-1])
    ax.set_xlabel("misclassification count")
    ax.set_title(f"Top {top_k} confused pairs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data/food-101")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--out_dir", default="eval/outputs")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True, parents=True)

    model, classes, device = load_classifier(args.checkpoint)
    _, _, test_loader, _ = build_loaders(args.data_root, batch_size=args.batch_size)

    res = evaluate(model, test_loader, device)
    cm = res.pop("confusion")

    tag = Path(args.checkpoint).stem
    (out / f"{tag}_metrics.json").write_text(json.dumps(res, indent=2))
    plot_confusion(cm, classes, out / f"{tag}_confusion_top20.png")
    np.save(out / f"{tag}_confusion.npy", cm)

    print(f"top-1={res['top1']:.4f} top-5={res['top5']:.4f} macro_f1={res['macro_f1']:.4f}")
    print(f"artifacts in {out}/{tag}_*")


if __name__ == "__main__":
    main()

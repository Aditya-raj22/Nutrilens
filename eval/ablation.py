"""Ablation study: frozen-vs-unfrozen backbone x augmentation on/off x architecture.

Produces a markdown table summarizing val accuracy for each cell. Assumes
checkpoints have been trained with matching tags by classifier/train.py.
"""
import json
from pathlib import Path

import pandas as pd


def scan_checkpoints(ckpt_dir="checkpoints"):
    rows = []
    for hist_file in Path(ckpt_dir).glob("*_history.json"):
        tag = hist_file.stem.replace("_history", "")
        hist = json.loads(hist_file.read_text())
        if not hist.get("val_acc"):
            continue
        best_val = max(hist["val_acc"])
        arch = "resnet50" if tag.startswith("resnet50") else \
               "vit_b_16" if tag.startswith("vit_b_16") else \
               "efficientnet_b3" if tag.startswith("efficientnet_b3") else "unknown"
        rows.append({
            "arch": arch,
            "frozen": "_frozen" in tag,
            "augmented": "_noaug" not in tag,
            "best_val_acc": round(best_val, 4),
            "epochs_trained": len(hist["val_acc"]),
            "tag": tag,
        })
    return pd.DataFrame(rows).sort_values(["arch", "frozen", "augmented"])


def main():
    df = scan_checkpoints()
    if df.empty:
        print("no checkpoints found in checkpoints/")
        return
    md = df.to_markdown(index=False)
    out = Path("eval/outputs/ablation.md")
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text("# Ablation Study\n\n" + md + "\n")
    print(md)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

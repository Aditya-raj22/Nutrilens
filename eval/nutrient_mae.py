"""Nutrient MAE on hand-weighed meal photos.

Expects data/hand_weighed/ with:
  - photos/ (jpg files)
  - labels.json: [{"photo": "01.jpg", "class": "pizza", "grams": 180,
                    "nutrients_ground_truth": {"Energy": 430, "Protein": 18, ...}}]
"""
import argparse
import json
from pathlib import Path

from PIL import Image
import numpy as np

from pipeline import infer_one, get_models


def mae(pred: dict, truth: dict, nutrients):
    errors = {}
    for n in nutrients:
        p = pred.get(n, 0)
        t = truth.get(n, 0)
        errors[n] = abs(p - t)
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--photos", default="data/hand_weighed")
    p.add_argument("--out", default="eval/outputs/nutrient_mae.json")
    args = p.parse_args()

    root = Path(args.photos)
    labels = json.loads((root / "labels.json").read_text())
    models = get_models()

    tracked = ["Energy", "Protein", "Iron, Fe", "Calcium, Ca",
               "Vitamin C, total ascorbic acid", "Fiber, total dietary"]
    all_errors = {n: [] for n in tracked}
    records = []

    for item in labels:
        img = Image.open(root / "photos" / item["photo"])
        result = infer_one(img, models=models)
        errs = mae(result["nutrients_scaled"], item.get("nutrients_ground_truth", {}), tracked)
        records.append({
            "photo": item["photo"],
            "true_class": item["class"],
            "pred_class": result["class"],
            "true_grams": item["grams"],
            "pred_grams": result["grams"],
            "errors": errs,
        })
        for n in tracked:
            all_errors[n].append(errs[n])

    summary = {n: {"mae": float(np.mean(v)), "median": float(np.median(v))}
               for n, v in all_errors.items()}
    gram_mae = float(np.mean([abs(r["true_grams"] - r["pred_grams"]) for r in records]))
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps({"gram_mae": gram_mae, "nutrient_mae": summary, "records": records}, indent=2))
    print(f"gram MAE: {gram_mae:.1f}")
    for n, s in summary.items():
        print(f"  {n}: MAE={s['mae']:.2f} median={s['median']:.2f}")


if __name__ == "__main__":
    main()

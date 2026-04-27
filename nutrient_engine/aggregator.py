"""Per-meal + per-day nutrient aggregation from USDA profiles."""
from collections import defaultdict


def scale_to_portion(per_100g: dict, grams: float):
    factor = grams / 100.0
    return {k: v * factor for k, v in per_100g.items()}


def sum_meals(meals):
    """meals: list of {class_name, grams, nutrients_per_100g}. Returns summed dict."""
    total = defaultdict(float)
    for m in meals:
        scaled = scale_to_portion(m["nutrients_per_100g"], m["grams"])
        for k, v in scaled.items():
            total[k] += v
    return dict(total)

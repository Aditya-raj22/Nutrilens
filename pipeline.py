"""End-to-end inference pipeline:

photo → classifier → SAM portion → USDA nutrients → deficiency flagger → LLM recs.

Cached object graph so repeated calls (Gradio app, eval) don't re-init models.
"""
from functools import lru_cache
from pathlib import Path
import json

from PIL import Image

from classifier.infer import load_classifier, predict_topk
from segmenter.sam_portion import load_sam, portion_for_image, TYPICAL_PORTION_G
from nutrient_engine.mapping import load_mapping, MAPPING_PATH
from nutrient_engine.usda_client import USDAClient
from nutrient_engine.aggregator import scale_to_portion, sum_meals
from user_profile.rda import personal_rda
from user_profile.deficiency import flag_deficiencies
from llm_client.openai_client import OpenAIClient


@lru_cache(maxsize=1)
def get_models(classifier_ckpt="checkpoints/vit_b_16_best.pt",
                sam_ckpt="checkpoints/sam_vit_b.pth"):
    model, classes, device = load_classifier(classifier_ckpt)
    try:
        predictor = load_sam(sam_ckpt, device=device)
    except Exception:
        predictor = None
    usda = USDAClient()
    llm = OpenAIClient()
    mapping = load_mapping(MAPPING_PATH)
    return model, classes, device, predictor, usda, llm, mapping


_nutrient_cache: dict = {}


def nutrients_for_class(usda, mapping, class_name):
    if class_name in _nutrient_cache:
        return _nutrient_cache[class_name]
    entry = mapping.get(class_name)
    if not entry or not entry.get("fdc_id"):
        return {}
    profile = usda.nutrient_profile(entry["fdc_id"])
    _nutrient_cache[class_name] = profile
    return profile


def _estimate_grams(image, class_name, predictor):
    """Use SAM if available; otherwise fall back to typical portion prior."""
    if predictor is not None:
        try:
            _, grams, score = portion_for_image(image, class_name, predictor=predictor)
            # Sanity check: reject implausible estimates
            typical = TYPICAL_PORTION_G.get(class_name, 200)
            if 0.5 * typical <= grams <= 4 * typical:
                return grams, score
        except Exception:
            pass
    # Fallback: typical portion with no score
    return float(TYPICAL_PORTION_G.get(class_name, 200)), 0.0


def infer_one(image: Image.Image, models=None):
    """Run classifier + SAM + nutrient scaling for a single meal image."""
    m = models or get_models()
    model, classes, device, predictor, usda, _llm, mapping = m
    topk = predict_topk(model, classes, image, device, k=3)
    top_class, top_prob = topk[0]
    grams, sam_score = _estimate_grams(image, top_class, predictor)
    per_100g = nutrients_for_class(usda, mapping, top_class)
    scaled = scale_to_portion(per_100g, grams)
    return {
        "top_predictions": topk,
        "class": top_class,
        "confidence": top_prob,
        "grams": grams,
        "sam_score": sam_score,
        "nutrients_per_100g": per_100g,
        "nutrients_scaled": scaled,
        "mask": None,
    }


def infer_day(images, user_profile, variant="few_shot_cot", models=None):
    """Full daily flow: list of images → per-meal breakdown + deficiencies + LLM recs."""
    m = models or get_models()
    _model, _classes, _device, _predictor, _usda, llm, _mapping = m

    meals = []
    for img in images:
        result = infer_one(img, models=m)
        meals.append({
            "class": result["class"],
            "grams": result["grams"],
            "nutrients_per_100g": result["nutrients_per_100g"],
        })

    total = sum_meals(meals)
    rda = personal_rda(**user_profile)
    gaps = flag_deficiencies(total, rda)
    recs = llm.recommend(gaps, user_profile, variant=variant) if gaps else "[]"
    return {
        "meals": meals,
        "total_nutrients": total,
        "rda": rda,
        "deficiencies": gaps,
        "recommendations": recs,
    }

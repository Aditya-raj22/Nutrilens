"""Zero-shot SAM + portion estimation.

Two calibration modes:
1. Credit card in frame: OpenCV detects the card (85.6x54mm), gives exact px→cm² scale.
2. No card: falls back to typical portion priors anchored by SAM mask fraction.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_G_PER_CM2 = 4.0
G_PER_CM2 = {
    "pizza": 2.8, "hamburger": 5.5, "steak": 7.0, "sushi": 3.5,
    "caesar_salad": 1.8, "french_fries": 2.2, "ice_cream": 3.2,
    "spaghetti_bolognese": 3.8, "chicken_wings": 4.5, "ramen": 3.5,
    "pho": 3.0, "chocolate_cake": 3.8, "pancakes": 3.0,
}

TYPICAL_PORTION_G = {
    "pizza": 450, "hamburger": 160, "steak": 225, "sushi": 180,
    "caesar_salad": 200, "french_fries": 120, "ice_cream": 130,
    "spaghetti_bolognese": 300, "chicken_wings": 200, "ramen": 450,
    "pho": 450, "chocolate_cake": 120, "pancakes": 200,
    "cheesecake": 120, "fried_rice": 280, "bibimbap": 350,
    "pad_thai": 300, "tacos": 180, "burrito": 300, "hot_dog": 100,
    "grilled_salmon": 180,
}

# Credit card ISO/IEC 7810 ID-1: 85.60 x 53.98 mm
CARD_W_MM, CARD_H_MM = 85.60, 53.98
CARD_AREA_CM2 = (CARD_W_MM / 10) * (CARD_H_MM / 10)  # 46.2 cm²
CARD_ASPECT = CARD_W_MM / CARD_H_MM  # ~1.586


def detect_card_cm2_per_px(image: Image.Image):
    """Return cm²/px calibration from credit card in image, or None if not found."""
    img = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    min_card_px = (w_img * h_img) * 0.01   # card must be ≥1% of image
    max_card_px = (w_img * h_img) * 0.40   # card must be ≤40% of image

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = cv2.contourArea(approx)
        if not (min_card_px < area < max_card_px):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if h == 0:
            continue
        aspect = w / h
        # Accept portrait or landscape orientation
        if not (abs(aspect - CARD_ASPECT) < 0.15 or abs(aspect - 1 / CARD_ASPECT) < 0.15):
            continue
        return CARD_AREA_CM2 / area  # cm² per pixel

    return None


def load_sam(checkpoint="checkpoints/sam_vit_b.pth", device=None):
    from segment_anything import sam_model_registry, SamPredictor
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint=str(Path(checkpoint)))
    sam.to(device=device)
    return SamPredictor(sam)


def centroid(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def predict_mask(predictor, image: Image.Image, point=None):
    img = np.array(image.convert("RGB"))
    predictor.set_image(img)
    h, w = img.shape[:2]
    if point is None:
        point = (w / 2, h / 2)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def estimate_grams(mask: np.ndarray, class_name, cm2_per_px=None):
    total_px = mask.size
    food_px = int(mask.sum())

    if cm2_per_px is not None:
        # Card calibration: exact scale known
        food_cm2 = food_px * cm2_per_px
        density = G_PER_CM2.get(class_name, DEFAULT_G_PER_CM2)
        return float(food_cm2 * density)
    else:
        # Fallback: typical portion anchored by SAM mask fraction
        food_frac = food_px / total_px
        typical_g = TYPICAL_PORTION_G.get(class_name, 200)
        scale = np.clip(food_frac / 0.30, 0.25, 3.0)
        return float(typical_g * scale)


def portion_for_image(image: Image.Image, class_name, predictor=None, point=None):
    """image + predicted class → (mask, grams, sam_confidence, calibrated)."""
    if predictor is None:
        predictor = load_sam()
    cm2_per_px = detect_card_cm2_per_px(image)
    mask, score = predict_mask(predictor, image, point=point)
    grams = estimate_grams(mask, class_name, cm2_per_px=cm2_per_px)
    return mask, grams, score

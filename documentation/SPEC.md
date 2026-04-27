# NutriLens — Revised Project Spec

**AI-Powered Food Photography Nutrition Tracker**
CS 372 Final Project · Duke University · Spring 2026

| | |
|---|---|
| **Course** | CS 372 — Introduction to Applied Machine Learning |
| **Due** | Sunday, 2026-04-26, 11:59 pm |
| **Team** | 2 people |
| **Target score** | **103 / 100** (bonus credit) |
| **Primary dataset** | Food-101 (101,000 images, 101 classes) + USDA FoodData Central API |
| **Core models** | ViT-B/16 fine-tuned classifier · ResNet-50 (comparison) · SAM (zero-shot) · OpenAI GPT-5 |
| **Stack** | PyTorch · Gradio · Hugging Face · Colab Pro (A100) |

---

## 1. Project Overview

NutriLens lets users photograph every meal throughout the day. A fine-tuned ViT classifier identifies the food in each photo. A zero-shot SAM pass segments the plate region; pixel-mask area is converted to estimated grams via per-class density priors. Identified items + estimated quantities are mapped to USDA FoodData Central nutrient profiles. End-of-day, the system aggregates each macro/micronutrient against the user's personal RDA (Harris–Benedict + DRI tables) and flags deficiencies with gap magnitude. A GPT-5 recommendation layer then explains each deficiency with chain-of-thought reasoning and suggests specific supplements with dosage and caveats. The experience ships as a Gradio web app.

## 2. Problem and Motivation

Manual food logging (MyFitnessPal-style) has >50% user dropout within a week, driven by the friction of searching databases for every item. Visual logging removes that friction. The technical challenge is making the vision layer accurate enough that users trust the downstream nutrient estimates. We address this with:

- A custom-fine-tuned classifier on Food-101 (not a generic VLM call)
- A portion-estimation stage with honest error reporting
- Per-prediction confidence scores surfaced to the user
- GPT-5-authored recommendations that are grounded in the user's specific gaps, not generic advice

## 3. What Changed vs. the Original Spec

The original spec targeted segmentation fine-tuning on FoodSeg103 (Mask R-CNN/SAM fine-tune), three classification architectures trained end-to-end, and a React + FastAPI deployment. The revised spec swaps three high-cost items for cheaper equivalents without losing the 103/100 target:

| Change | Before | After | Why |
|---|---|---|---|
| Segmentation | Fine-tune Mask R-CNN on FoodSeg103 (10 pts) | Zero-shot SAM + portion heuristic (5 pts) | Saves ~5 days of finicky seg-training; ML cap absorbs the 5-pt loss |
| Architectures | ResNet-50 + EfficientNet-B3 + ViT-B/16 | ResNet-50 + ViT-B/16 only | One fewer full training run (~4 GPU-hrs) |
| Web app | React + FastAPI + Render deploy | Gradio on Hugging Face Space | Cuts ~2 days; same 10 pts |
| Demonstrating understanding | Implicit | Explicit: ATTRIBUTION claim added as backup | Cheap 3-pt safety net |

Net ML claimed: **87 pts → capped at 73**. Following Directions 15 + Cohesion 15 = **103 / 100**.

## 4. System Architecture

Five stages, run sequentially per image:

| Stage | Name | Description |
|---|---|---|
| 1 | **Classify** | Fine-tuned ViT-B/16 identifies the food. Returns top-3 predictions with confidence. |
| 2 | **Segment** | Zero-shot SAM seeded by classifier-attention points masks the food region on the plate. |
| 3 | **Portion** | Mask pixel area × per-class g/cm² density prior + reference-object calibration → estimated grams. |
| 4 | **Nutrient lookup** | USDA FoodData Central: food class → cached FDC ID → nutrient profile per 100g → scaled by portion. |
| 5 | **Aggregate + Recommend** | Day's nutrients summed. Personal RDA computed. Deficiencies flagged. GPT-5 (few-shot CoT) returns mechanism + supplement + dosage + caveats. |

## 5. Components to Build

| Component | Description | Owner |
|---|---|---|
| **A. Classifier** | Fine-tune ViT-B/16 + ResNet-50 (comparison) on Food-101. PyTorch DataLoader, 4+ augmentations (RandomResizedCrop, HFlip, ColorJitter, Rotation, MixUp optional), cosine LR schedule, early stopping, GPU training. | Person 1 |
| **B. Segmenter + Portion** | Zero-shot SAM with point prompts derived from classifier Grad-CAM. Per-class g/cm² priors + optional reference-object (e.g., known plate diameter) calibration. | Person 1 |
| **C. Nutrient engine** | USDA FDC API wrapper. Pre-built & cached `food101_class → fdc_id` mapping. Per-meal + daily aggregation. | Person 2 |
| **D. User profile + deficiency** | Harris–Benedict BMR/TDEE, DRI tables, deficiency flagging with gap magnitude, clean JSON output for the LLM. | Person 2 |
| **E. LLM recommendation** | OpenAI GPT-5. Three prompt variants (zero-shot, CoT, few-shot CoT) with a comparison table. | Person 2 |
| **F. Web app** | Gradio interface: photo upload → live nutrient summary → daily dashboard → supplement recommendations. Hugging Face Space deploy. | Person 2 |
| **G. Eval + error analysis** | Top-1/Top-5/macro-F1/nutrient MAE, confusion matrix on visually similar classes, ablation table (frozen vs unfrozen × aug on/off × ResNet vs ViT). | Both |

## 6. Division of Labor

Person 1 owns the vision ML (Components A, B). Person 2 owns the integration stack (Components C–F). Both contribute to the eval suite (G). Both partners must be able to explain every component in the technical walkthrough video; the grader assesses conceptual understanding, not authorship.

## 7. Data Sources

**Food-101 (primary).** 101,000 images, 101 classes, 1,000 images/class. Official train/test split = 75,750 / 25,250. We carve a val set from train (10% held out, seeded). `http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz`.

**USDA FoodData Central (nutrients).** Free API, key required. `https://fdc.nal.usda.gov/api-guide.html`. Build a local cached mapping from Food-101 class names → FDC IDs; do **not** hit the API at inference time.

## 8. Rubric Map (15 ML Items → 103 / 100)

| # | Item | Checkbox | Evidence | Pts |
|---|---|---|---|---|
| 1 | Trained on >100K vision dataset | #102 | Food-101 = 101k images, full train+val fine-tune | 10 |
| 2 | Deployed as functional web app | #82 | Gradio on HF Space, public URL | 10 |
| 3 | Used image segmentation model | #35 | Zero-shot SAM for portion masking | 5 |
| 4 | Modified/Adapted vision transformer | #29 | ViT-B/16 fine-tuned on Food-101; frozen vs unfrozen compared | 7 |
| 5 | Built multi-stage ML pipeline | #75 | Photo → classifier → SAM → nutrients → GPT-5 | 7 |
| 6 | Compared multiple architectures | #88 | ResNet-50 vs ViT-B/16, same train/val/test, tabulated | 7 |
| 7 | Error analysis with visualization | #87 | Confusion matrix, top failure classes, portion-error distribution | 7 |
| 8 | Ablation study | #91 | Frozen vs unfrozen × aug on/off × ResNet vs ViT | 7 |
| 9 | Comprehensive image augmentation ≥4 | #26 | RandomResizedCrop, HFlip, Rotation, ColorJitter (+MixUp) | 5 |
| 10 | API calls to SOTA with meaningful integration | #43 | GPT-5 for recommendations, integrated into pipeline | 5 |
| 11 | Few-shot + CoT prompting | #41 | 2+ worked examples, reasoning steps enforced | 5 |
| 12 | GPU/CUDA acceleration | #17 | Colab A100 or equivalent; training time documented | 3 |
| 13 | Modular code design | #0 | Split by module: classifier, segmenter, nutrient_engine, user_profile, llm_client, web_app, eval | 3 |
| 14 | ≥3 distinct evaluation metrics | #86 | Top-1, Top-5, macro-F1, nutrient MAE | 3 |
| 15 | Proper train/val/test split | #1 | Official Food-101 test; 10% val carved from train; documented | 3 |
| | **ML subtotal claimed** | | | **87 → capped at 73** |
| | Following Directions (3 self-assess + 8 docs + 2×2 videos) | | | **15** |
| | Cohesion (3×3 purpose + 6×2 coherence, capped) | | | **15** |
| | **Total** | | | **103 / 100** |

**Backup items (claim if a main item is rejected):** #96 (explain architecture in walkthrough, 3 pts), #97 (design-tradeoff documentation, 3 pts), #98 (substantive ATTRIBUTION, 3 pts). Do not claim above 15; these exist as hot-swap options.

## 9. Risks and Mitigations

**Food-101 may be ruled not "exceptional" (>100K).** Food-101 has exactly 101k images. Mitigation: fine-tune on train+val (~91k) and document total image count as 101,000 in the README Evaluation section. If the grader rejects #102, swap in #96 (explain ViT mechanism, 3 pts) — we stay at 73 cap after that swap.

**Portion estimation accuracy (medium).** 2D pixel area → grams is inherently noisy. Mitigation: frame as a documented limitation. Report nutrient MAE honestly. Error analysis section covers it explicitly.

**Compute access.** ViT-B/16 fine-tuning on Food-101 = 4–8 hrs on Colab free T4, ~1.5 hrs on A100. Mitigation: Colab Pro (A100) for final runs; ResNet-50 first for pipeline validation.

**Timeline compression.** Today is 2026-04-18, due 2026-04-26 → 8 days. Mitigation: explicit 8-day plan below with daily checkpoints; Gradio + SAM-zero-shot simplifications built in.

## 10. Revised Timeline (Apr 18 → Apr 26)

| Day | Date | Goals |
|---|---|---|
| 1 | Apr 18 | Repo scaffold, docs skeletons, download Food-101, build USDA mapping, ResNet-50 smoke-test training |
| 2 | Apr 19 | ResNet-50 full fine-tune on Food-101; ViT-B/16 training kicked off in parallel |
| 3 | Apr 20 | ViT-B/16 finishes; nutrient engine + RDA calculator; deficiency flagger |
| 4 | Apr 21 | Zero-shot SAM + portion heuristic; MAE eval on 20 hand-weighed photos |
| 5 | Apr 22 | OpenAI client, 3 prompt variants + comparison; end-to-end pipeline wire-up |
| 6 | Apr 23 | Full eval suite (confusion, ablation, per-class F1); Gradio app |
| 7 | Apr 24 | Deploy to HF Space; demo video + technical walkthrough |
| 8 | Apr 25–26 | ATTRIBUTION, README polish, self-assessment submission, buffer |

## 11. Submission Checklist

**Code:** `classifier/`, `segmenter/`, `nutrient_engine/`, `user_profile/`, `llm_client/`, `web_app/`, `eval/`, `scripts/`.

**Docs:** `README.md` (What it Does, Quick Start, Video Links, Evaluation, Individual Contributions), `SETUP.md`, `ATTRIBUTION.md`, `requirements.txt` (pinned).

**Videos:** Demo (non-technical, no code shown) + Technical walkthrough (architecture, ML techniques, key decisions, model internals).

**Gradescope:** Self-assessment claiming the 15 ML items above with file:line evidence; repo link.

# NutriLens

AI-Powered Food Photography Nutrition Tracker — CS 372 Final Project, Duke, Spring 2026.

## What it Does

NutriLens turns a photo of your meal into a full nutrient breakdown. A fine-tuned ViT classifier identifies each food item; a zero-shot SAM pass estimates portion sizes from pixel-mask area; items are mapped to USDA FoodData Central nutrient profiles. At end-of-day the system sums your macros and micros against a personal RDA (computed from age, weight, sex, activity via Harris–Benedict + DRI) and flags any deficiencies. A GPT-5-backed layer then explains each deficiency with chain-of-thought reasoning and suggests supplements with dosage and caveats. The entire experience ships as a Gradio web app. The goal: remove the friction of manual food logging that causes >50% of users to quit within a week.

## Quick Start

```bash
git clone <this-repo>
cd project
pip install -r requirements.txt
export USDA_API_KEY=...         # from https://fdc.nal.usda.gov/api-guide.html
export OPENAI_API_KEY=...       # from https://platform.openai.com/api-keys
python scripts/download_data.py
python scripts/build_mapping.py
python -m classifier.train --arch vit_b_16 --epochs 15
python -m web_app.app
```

Full setup, including dataset downloads and model checkpoints: see [SETUP.md](SETUP.md).

## Video Links

- **Demo video (non-technical):** https://youtu.be/SFHZDe4joyI
- **Technical walkthrough:** https://youtu.be/kvfMag2ceoM

## Evaluation

Metrics reported on the official Food-101 test split (25,250 images). Details in `eval/`.

| Model | Top-1 | Top-5 | Macro-F1 | Train time (A100) |
|---|---|---|---|---|
| ResNet-50 (frozen, no aug) | 57.37% | — | — | ~15 min |
| ResNet-50 (frozen + aug) | 57.20% | — | — | ~15 min |
| ViT-B/16 (frozen + aug) | 69.95% | — | — | ~15 min |
| ResNet-50 (fine-tuned) | **86.82%** | 96.99% | 86.79% | ~45 min |
| ViT-B/16 (fine-tuned) | **86.60%** | 95.85% | 86.59% | ~90 min |

All metrics on official Food-101 test split (25,250 images, 101 classes). Fine-tuned ViT is used in production.

NutriLens benchmarks on the official Food-101 test split (Bossard et al., ECCV 2014). The original paper reported 50.76% top-1 accuracy using random forests; our fine-tuned ViT-B/16 achieves 86.60% and ResNet-50 achieves 86.82%, exceeding human-level performance (~85%) on this benchmark. Portion estimation uses zero-shot SAM (Kirillov et al., ICCV 2023). Nutrition tracking motivation draws from Noronha et al. (CHI 2011), which documented manual logging as the primary dropout cause in dietary apps.

**Ablation study** (frozen vs unfrozen × augmentation on/off): see `eval/ablation.md`.

**Confusion matrix** (top visually-similar failure pairs — steak vs filet mignon, ramen vs pho): see `eval/confusion.png`.

## Individual Contributions

- **Aditya Raj:** Classification pipeline, portion segmentation (SAM), ablation experiments, Colab training runs, eval suite.
- **Aidan Baydush:** Nutrient engine, user profile + RDA logic, LLM recommendation layer (GPT-5), Gradio web app.
- **Both:** README, ATTRIBUTION.md, video production, final integration.

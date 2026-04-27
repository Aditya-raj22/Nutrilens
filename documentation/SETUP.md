# SETUP

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required. CUDA 11.8+ recommended for training (falls back to CPU for inference).

## 2. API Keys

Set the following environment variables (or put them in a `.env` file and `source` it):

```bash
export USDA_API_KEY=...        # https://fdc.nal.usda.gov/api-guide.html — free
export OPENAI_API_KEY=...      # https://platform.openai.com/api-keys
```

## 3. Download Data

```bash
python scripts/download_data.py
```

This downloads **Food-101** (~5 GB) to `data/food-101/`. Expect 10–20 min on a fast connection.

## 4. Build the USDA Mapping

```bash
python scripts/build_mapping.py
```

Queries USDA FDC once per Food-101 class (101 calls) and caches to `data/food101_fdc_map.json`. Run once; committed to the repo after the first run.

## 5. Train the Classifier

```bash
# Baseline (fast, for pipeline validation)
python -m classifier.train --arch resnet50 --epochs 15 --batch_size 64

# Full run (the final submission)
python -m classifier.train --arch vit_b_16 --epochs 20 --batch_size 32
```

Checkpoints write to `checkpoints/{arch}_best.pt`. Training history (loss/acc per epoch) writes to `checkpoints/{arch}_history.json`.

## 6. Download SAM Weights (Segmentation)

```bash
python scripts/download_sam.py
```

Downloads the ViT-B SAM checkpoint (~375 MB) to `checkpoints/sam_vit_b.pth`.

## 7. Run the App

```bash
python -m web_app.app
```

Gradio launches on `http://localhost:7860`. Uploading a photo triggers the full inference pipeline.

## 8. Deploy (optional)

```bash
# Hugging Face Space deployment
huggingface-cli login
huggingface-cli repo create nutrilens --type space --space_sdk gradio
git push hf main
```

## 9. Run Evaluation

```bash
python -m eval.run --arch vit_b_16          # top-1, top-5, macro-F1, confusion matrix
python -m eval.ablation                     # ablation table
python -m eval.nutrient_mae --photos data/hand_weighed  # portion-error MAE
```

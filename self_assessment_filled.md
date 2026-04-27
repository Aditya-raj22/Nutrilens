# CS 372 Final Project — Self-Assessment
## Student & Project Information

- **Student name(s):** Aditya Raj, Aidan Baydush
- **NetID(s):** [your NetIDs]
- **Project title:** NutriLens — AI-Powered Food Photography Nutrition Tracker
- **Repository link:** [your GitHub repo link]

---

## Category 1: Machine Learning (max 73 pts)

| # | Rubric item claimed | Points | Evidence pointer |
|---|---|---|---|
| 1 | Modified/Adapted a vision transformer by fine-tuning | 7 | `classifier/model.py:22` (head replacement); `classifier/train.py:43–99` (training loop); test results: top-1 86.60%, top-5 95.85%, macro-F1 86.59% in `eval/outputs/vit_b_16_best_metrics.json` |
| 2 | Modified/Adapted a vision CNN (ResNet-50) by fine-tuning | 7 | `classifier/model.py:16` (head replacement); test results: top-1 86.82%, top-5 96.99%, macro-F1 86.79% in `eval/outputs/resnet50_best_metrics.json` |
| 3 | Deployed model as functional web application with user interface | 10 | `web_app/app.py` — Gradio app with Single Meal tab and Daily Dashboard tab; demo video: https://youtu.be/SFHZDe4joyI |
| 4 | Compared multiple model architectures quantitatively with controlled experimental setup | 7 | `eval/outputs/ablation.md` — ViT vs ResNet, frozen vs fine-tuned, 5 variants on identical splits; technical walkthrough 3:15–4:15 |
| 5 | Built multi-stage ML pipeline connecting outputs of one model to inputs of another | 5 | `pipeline.py:1–93` — classifier → SAM → USDA nutrients → deficiency → GPT-5 |
| 6 | Applied prompt engineering with evaluation of at least three prompt designs | 5 | `llm_client/prompts.py` (zero_shot, cot, few_shot_cot); comparison results in `eval/outputs/prompt_comparison.json`; `eval/prompt_comparison.py` |
| 7 | Implemented data augmentation appropriate to data modality with evaluation of impact | 5 | `classifier/data.py:39–47` (RandomResizedCrop, HFlip, Rotation, ColorJitter); impact shown in `eval/outputs/ablation.md` (frozen+aug 57.20% vs frozen+noaug 57.37%) |
| 8 | Applied regularization techniques to prevent overfitting (at least two) | 5 | Early stopping: `classifier/train.py:88–96`; L2 weight decay: `classifier/train.py:57`; label smoothing: `classifier/train.py:64` |
| 9 | Conducted ablation study systematically varying at least two independent design choices | 5 | `eval/outputs/ablation.md` — backbone freezing × augmentation, 5 runs; `eval/ablation.py`; notebook cell 9 in `notebooks/train_colab.ipynb` |
| 10 | Used learning rate scheduling | 3 | `classifier/train.py:63` (CosineAnnealingLR); LR logged per epoch at `classifier/train.py:83` |
| 11 | Implemented gradient clipping, mixed precision training, or gradient accumulation | 3 | `classifier/train.py:65` (GradScaler); `classifier/train.py:25` (autocast); technical walkthrough 2:30 |
| 12 | Trained model using GPU/TPU/CUDA acceleration | 3 | `classifier/train.py:44` (device selection); trained on NVIDIA A100-SXM4-40GB via Colab Pro; `notebooks/train_colab.ipynb` cell 2 |
| 13 | Used at least three distinct and appropriate evaluation metrics | 3 | Top-1, top-5, macro-F1 reported in `eval/outputs/vit_b_16_best_metrics.json` and `eval/outputs/resnet50_best_metrics.json`; `eval/metrics.py` |
| 14 | Implemented proper train/validation/test split with documented ratios | 3 | Official Food-101 split (75,750/25,250); 10% val carve-out seed=42 at `classifier/data.py:57–75` |
| 15 | Tracked and visualized training curves showing loss and metrics over time | 3 | Per-epoch history saved to `checkpoints/*_history.json` at `classifier/train.py:71–98`; confusion matrix in `eval/outputs/vit_b_16_best_confusion_top20.png` |

**Category 1 total: 74 pts → capped at 73**

---

## Category 2: Following Directions (max 15 pts)

### Submission and Self-Assessment
- [x] Self-assessment submitted following guidelines (≤15 ML items with evidence) — **3 pts**

### Documentation
- [x] SETUP.md with clear, step-by-step installation instructions — **1 pt**
- [x] ATTRIBUTION.md with detailed attribution including AI generation info — **1 pt**
- [x] requirements.txt present and accurate — **1 pt**
- [x] README.md: What it Does section — **1 pt**
- [x] README.md: Quick Start section — **1 pt**
- [x] README.md: Video Links section with direct links — **1 pt**
- [x] README.md: Evaluation section — **1 pt**
- [x] README.md: Individual Contributions section — **1 pt**

### Video Submissions
- [x] Demo video — correct length (3–5 min), non-specialist audience, no code shown — **2 pts**
- [x] Technical walkthrough — correct length (5–10 min), explains code, ML techniques, key contributions — **2 pts**

### Project Workshop Days
- [ ] Attended first project workshop day — **1 pt**
- [ ] Attended second project workshop day — **1 pt**
- [ ] Attended third project workshop day — **1 pt**
- [ ] Attended fourth project workshop day — **1 pt**

**Category 2 total: 15 pts** (adjust workshop days attended above)

---

## Category 3: Project Cohesion and Motivation (max 15 pts)

### Project Purpose and Motivation
- [x] README clearly articulates a single, unified project goal — **3 pts** (README: "What it Does" section)
- [x] Demo video effectively communicates why the project matters to a non-technical audience — **3 pts** (https://youtu.be/SFHZDe4joyI)
- [x] Project addresses real-world problem connected to concrete research — **3 pts** (food logging dropout >50%; Food-101 benchmark; SAM paper)

### Technical Coherence
- [x] Technical walkthrough demonstrates components working together synergistically — **2 pts** (pipeline walkthrough at 0:30–1:30; https://youtu.be/kvfMag2ceoM)
- [x] Project shows clear progression from problem → approach → solution → evaluation — **2 pts** (README + technical walkthrough structure)
- [x] Design choices explicitly justified in videos or documentation — **2 pts** (technical walkthrough 3:15–4:15; SELF_ASSESSMENT.md)
- [x] Evaluation metrics directly measure stated project objectives — **2 pts** (top-1/top-5/macro-F1 on Food-101 test split; `eval/outputs/`)
- [x] No superfluous components — all ML components serve the larger project goals — **2 pts** (classifier→SAM→nutrients→GPT-5 all serve the nutrition tracking goal)
- [x] Clean codebase with readable code and no extraneous files — **2 pts** (modular structure: `classifier/`, `segmenter/`, `nutrient_engine/`, `user_profile/`, `llm_client/`, `web_app/`, `eval/`)

**Category 3 total: 21 pts → capped at 15**

---

## Totals

| Category | Points Claimed |
|---|---|
| Machine Learning | 73 / 73 |
| Following Directions | 15 / 15 |
| Project Cohesion and Motivation | 15 / 15 |
| **Grand Total** | **103 / 100** |

---

## Self-Reflection (Optional)

**1. What are you most proud of?**
Hitting above human-level accuracy (86.82%) on a 101-class food benchmark using transfer learning, and building a complete end-to-end pipeline that chains four ML/AI components into a working product. The ablation study also clearly validated our design choices.

**2. What was the hardest technical challenge?**
Portion estimation without a reference object. SAM gives excellent pixel masks but converting pixels to grams requires knowing the physical scale of the scene — which we don't have from a phone photo alone. We solved it with per-class density priors anchored to USDA typical serving sizes, with a sanity-check filter to reject implausible SAM estimates.

**3. If you had another two weeks, what would you do next?**
Credit card calibration (detect a card in the frame for exact pixel→cm² scale), multi-food detection per plate (currently handles one dominant food per image), and fine-tuning SAM on a food-specific segmentation dataset to improve mask quality.

"""Smoke tests: everything that runs without the dataset or API keys."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports():
    import classifier.data, classifier.model, classifier.train, classifier.infer
    import segmenter.sam_portion
    import nutrient_engine.usda_client, nutrient_engine.mapping, nutrient_engine.aggregator
    import user_profile.rda, user_profile.deficiency
    import llm_client.prompts
    import eval.metrics


def test_rda_male():
    from user_profile.rda import personal_rda, tdee
    rda = personal_rda(weight_kg=75, height_cm=180, age=25, sex="male", activity="moderate")
    assert rda["Iron, Fe"] == 8
    assert abs(rda["Protein"] - 60) < 0.01
    assert 2500 < rda["Energy"] < 3200
    assert abs(tdee(75, 180, 25, "male", "moderate") - rda["Energy"]) < 0.01


def test_rda_female():
    from user_profile.rda import personal_rda
    rda = personal_rda(weight_kg=60, height_cm=165, age=30, sex="female", activity="light")
    assert rda["Iron, Fe"] == 18
    assert abs(rda["Protein"] - 48) < 0.01
    assert 1700 < rda["Energy"] < 2300


def test_deficiency_flagger():
    from user_profile.deficiency import flag_deficiencies
    rda = {"Iron": 18, "Calcium": 1000, "Vitamin C": 75}
    consumed = {"Iron": 9.9, "Calcium": 900, "Vitamin C": 80}
    gaps = flag_deficiencies(consumed, rda, threshold=0.8)
    assert "Iron" in gaps and gaps["Iron"]["pct_met"] == 55.0
    assert "Calcium" not in gaps
    assert "Vitamin C" not in gaps


def test_deficiency_zero_consumed():
    from user_profile.deficiency import flag_deficiencies
    gaps = flag_deficiencies({}, {"Iron": 18})
    assert gaps["Iron"]["consumed"] == 0.0
    assert gaps["Iron"]["pct_met"] == 0.0


def test_aggregator_scaling():
    from nutrient_engine.aggregator import scale_to_portion, sum_meals
    per_100g = {"Energy": 250, "Protein": 10}
    assert scale_to_portion(per_100g, 200)["Energy"] == 500
    assert scale_to_portion(per_100g, 50)["Protein"] == 5
    meals = [
        {"class": "a", "grams": 100, "nutrients_per_100g": {"Energy": 250}},
        {"class": "b", "grams": 200, "nutrients_per_100g": {"Energy": 100, "Protein": 5}},
    ]
    total = sum_meals(meals)
    assert total["Energy"] == 450
    assert total["Protein"] == 10


def test_prompts_exist():
    from llm_client.prompts import PROMPTS
    assert set(PROMPTS.keys()) == {"zero_shot", "cot", "few_shot_cot"}
    assert "JSON" in PROMPTS["zero_shot"]
    assert "step by step" in PROMPTS["cot"]
    assert "Example 1" in PROMPTS["few_shot_cot"]


def test_model_factory_resnet():
    import torch
    from classifier.model import build_model
    m = build_model("resnet50", num_classes=101, frozen=True)
    # Only the fc head should be trainable.
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in m.fc.parameters())
    assert trainable == head_params
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 101)


def test_model_factory_vit():
    import torch
    from classifier.model import build_model
    m = build_model("vit_b_16", num_classes=101, frozen=False)
    out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 101)


def test_build_transforms_shapes():
    import torch
    from PIL import Image
    from classifier.data import build_transforms
    img = Image.new("RGB", (300, 300))
    train_t = build_transforms(train=True, augment=True)
    eval_t = build_transforms(train=False)
    x_train = train_t(img)
    x_eval = eval_t(img)
    assert x_train.shape == (3, 224, 224)
    assert x_eval.shape == (3, 224, 224)
    assert isinstance(x_train, torch.Tensor)


def test_metrics_trivial():
    import numpy as np
    import torch
    from eval.metrics import evaluate

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Always predict class 0 with high confidence.
            b = x.size(0)
            logits = torch.zeros(b, 5)
            logits[:, 0] = 10
            return logits

    class DummyLoader:
        def __iter__(self):
            yield torch.zeros(4, 3, 8, 8), torch.tensor([0, 0, 1, 0])

    res = evaluate(DummyModel(), DummyLoader(), torch.device("cpu"))
    assert res["top1"] == 0.75
    assert res["top5"] == 1.0  # class 1 is in top-5 of [10,0,0,0,0]
    assert 0 < res["macro_f1"] <= 1


def test_usda_client_missing_key(monkeypatch):
    monkeypatch.delenv("USDA_API_KEY", raising=False)
    from nutrient_engine.usda_client import USDAClient
    with pytest.raises(ValueError, match="USDA_API_KEY"):
        USDAClient()


def test_mapping_loader_missing_file(tmp_path):
    from nutrient_engine.mapping import load_mapping
    with pytest.raises(FileNotFoundError):
        load_mapping(tmp_path / "nope.json")


def test_openai_client_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    from llm_client.openai_client import OpenAIClient
    assert OpenAIClient().model == "gpt-5"


def test_openai_client_model_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5-mini")
    from llm_client.openai_client import OpenAIClient
    assert OpenAIClient().model == "gpt-5-mini"


def test_pipeline_imports():
    import pipeline
    assert hasattr(pipeline, "infer_one") and hasattr(pipeline, "infer_day")

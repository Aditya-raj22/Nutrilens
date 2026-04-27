"""Inference wrapper: load checkpoint, predict top-k on a PIL image."""
import torch
import torch.nn.functional as F

from .data import build_transforms
from .model import build_model


def load_classifier(checkpoint_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt["arch"], num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"], device


@torch.no_grad()
def predict_topk(model, classes, image, device, k=3):
    tf = build_transforms(train=False)
    x = tf(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]
    top_p, top_i = probs.topk(k)
    return [(classes[i], float(p)) for p, i in zip(top_p.cpu(), top_i.cpu())]

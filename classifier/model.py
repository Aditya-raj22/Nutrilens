"""Model factory: ResNet-50, ViT-B/16, EfficientNet-B3 with custom heads."""
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vit_b_16, ViT_B_16_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
)


def build_model(arch="resnet50", num_classes=101, frozen=False):
    if arch == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if frozen:
            for p in m.parameters():
                p.requires_grad = False
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        if frozen:
            for p in m.parameters():
                p.requires_grad = False
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif arch == "efficientnet_b3":
        m = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        if frozen:
            for p in m.parameters():
                p.requires_grad = False
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"unknown arch: {arch}")
    return m

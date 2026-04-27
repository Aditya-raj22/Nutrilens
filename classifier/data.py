"""Food-101 dataset, transforms, and DataLoaders.

Uses the official train/test split from meta/{train,test}.txt.
A 10% validation subset is carved from the official train split (seeded).
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


class Food101Dataset(Dataset):
    def __init__(self, root, split="train", transform=None):
        root = Path(root)
        meta_file = root / "meta" / f"{split}.txt"
        lines = meta_file.read_text().splitlines()
        self.samples = [(root / "images" / f"{p}.jpg", p.split("/")[0]) for p in lines if p]
        classes_file = root / "meta" / "classes.txt"
        self.classes = [c for c in classes_file.read_text().splitlines() if c]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[cls]


def build_transforms(train=True, augment=True):
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def build_loaders(root, batch_size=64, augment=True, val_frac=0.1, seed=42, num_workers=4):
    train_aug = Food101Dataset(root, "train", build_transforms(train=True, augment=augment))
    train_clean = Food101Dataset(root, "train", build_transforms(train=False))
    test_set = Food101Dataset(root, "test", build_transforms(train=False))

    n = len(train_aug)
    n_val = int(n * val_frac)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train = Subset(train_aug, train_idx)
    val = Subset(train_clean, val_idx)

    common = dict(num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, **common),
        DataLoader(val, batch_size=batch_size, shuffle=False, **common),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, **common),
        train_aug.classes,
    )

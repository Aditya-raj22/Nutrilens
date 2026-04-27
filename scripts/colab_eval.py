"""Run from /content/project after Drive is mounted. Copies checkpoints and runs eval."""
import os
import shutil
import subprocess
import sys

DRIVE = "/content/drive/MyDrive/NutriLens"
PROJECT = "/content/project"

# Copy checkpoints
src = f"{DRIVE}/checkpoints"
dst = f"{PROJECT}/checkpoints"
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
print("checkpoints:", os.listdir(dst))

# Copy mapping if missing
mapping_dst = f"{PROJECT}/data/food101_fdc_map.json"
mapping_src = f"{DRIVE}/data/food101_fdc_map.json"
if not os.path.exists(mapping_dst) and os.path.exists(mapping_src):
    os.makedirs(os.path.dirname(mapping_dst), exist_ok=True)
    shutil.copy(mapping_src, mapping_dst)
    print("copied mapping")

os.chdir(PROJECT)

# Run eval
for ckpt in ["checkpoints/vit_b_16_best.pt", "checkpoints/resnet50_best.pt"]:
    if os.path.exists(ckpt):
        print(f"\n=== eval {ckpt} ===")
        subprocess.run([sys.executable, "-m", "eval.run", "--checkpoint", ckpt], check=True)

# Ablation table
print("\n=== ablation ===")
subprocess.run([sys.executable, "-m", "eval.ablation"], check=True)

# Persist eval outputs back to Drive
eval_dst = f"{DRIVE}/eval/outputs"
os.makedirs(eval_dst, exist_ok=True)
eval_src = f"{PROJECT}/eval/outputs"
if os.path.exists(eval_src):
    for f in os.listdir(eval_src):
        shutil.copy(os.path.join(eval_src, f), os.path.join(eval_dst, f))
    print(f"saved eval outputs to Drive")

"""Download Food-101 dataset."""
import subprocess
from pathlib import Path


def download_food101(root="data"):
    root = Path(root)
    root.mkdir(exist_ok=True)
    target = root / "food-101"
    if target.exists() and (target / "meta" / "train.txt").exists():
        print(f"food-101 already at {target}")
        return
    tar = root / "food-101.tar.gz"
    if not tar.exists():
        url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        print(f"downloading {url} (~5GB, 10-20 min)...")
        subprocess.run(["curl", "-L", url, "-o", str(tar)], check=True)
    print(f"extracting to {root}...")
    subprocess.run(["tar", "-xzf", str(tar), "-C", str(root)], check=True)
    print(f"food-101 ready at {target}")


if __name__ == "__main__":
    download_food101()

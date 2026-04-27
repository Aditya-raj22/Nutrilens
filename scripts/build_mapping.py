"""Build the Food-101 → USDA FDC ID mapping once and cache to data/food101_fdc_map.json.

Run after `scripts/download_data.py` (so Food-101 class list is available on disk),
or pass --classes_file to point at a class list manually.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nutrient_engine.mapping import build_mapping
from nutrient_engine.usda_client import USDAClient


def load_classes(classes_file):
    p = Path(classes_file)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run scripts/download_data.py first, or pass --classes_file."
        )
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes_file", default="data/food-101/meta/classes.txt"
    )
    parser.add_argument("--out", default="data/food101_fdc_map.json")
    args = parser.parse_args()

    classes = load_classes(args.classes_file)
    print(f"loaded {len(classes)} classes")
    client = USDAClient()
    build_mapping(client, classes, out_path=args.out)
    print(f"done; wrote {args.out}")


if __name__ == "__main__":
    main()

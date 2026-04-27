"""Download SAM ViT-B checkpoint."""
import subprocess
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def main(out="checkpoints/sam_vit_b.pth"):
    out = Path(out)
    out.parent.mkdir(exist_ok=True, parents=True)
    if out.exists():
        print(f"{out} already exists")
        return
    print(f"downloading SAM ViT-B (~375 MB)...")
    subprocess.run(["curl", "-L", URL, "-o", str(out)], check=True)
    print(f"SAM checkpoint ready at {out}")


if __name__ == "__main__":
    main()

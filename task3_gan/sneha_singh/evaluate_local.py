"""Local CycleGAN metrics (FID/KID/LPIPS/cycle L1, etc.).

Fill after Part 3 training. Reads images from outputs/pred_A2B and pred_B2A.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    print("member folder:", root.name)
    print("pred_A2B:", (root / "outputs" / "pred_A2B").exists())
    print("pred_B2A:", (root / "outputs" / "pred_B2A").exists())
    raise NotImplementedError("Wire FID/KID/LPIPS after the model exists.")


if __name__ == "__main__":
    main()

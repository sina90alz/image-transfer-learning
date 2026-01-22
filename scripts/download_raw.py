from __future__ import annotations

import shutil
from pathlib import Path
import kagglehub


def find_class_folders(root: Path) -> dict[str, Path]:
    """
    Tries to find class folders in the downloaded dataset.
    Supports common structures:
      - root/train/NORMAL, root/train/PNEUMONIA, root/train/TUBERCULOSIS (or test/val)
      - root/NORMAL, root/PNEUMONIA, root/TUBERCULOSIS
      - root/chest_xray/train/NORMAL, etc.
    Returns mapping: {"NORMAL": path, "PNEUMONIA": path, "TUBERCULOSIS": path}
    """
    # Common class names for chest X-ray datasets
    class_names = ["NORMAL", "PNEUMONIA","TUBERCULOSIS"]

    # 1) If dataset has train/val/test structure, use train first
    for split in ["train", "training", "Train", "TRAIN"]:
        split_dir = root / split
        if split_dir.exists():
            mapping = {c: split_dir / c for c in class_names}
            if all(p.exists() for p in mapping.values()):
                return mapping

    # 2) If classes exist at root
    mapping = {c: root / c for c in class_names}
    if all(p.exists() for p in mapping.values()):
        return mapping

    # 3) Search anywhere under root (one level deep is often enough, but we’ll be robust)
    found = {}
    for c in class_names:
        matches = list(root.rglob(c))
        # pick the first directory match
        matches = [m for m in matches if m.is_dir()]
        if matches:
            found[c] = matches[0]

    if len(found) == 2:
        return found

    raise RuntimeError(
        f"Could not locate class folders ({class_names}) under downloaded dataset.\n"
        f"Downloaded root: {root}\n"
        f"Top-level contents: {list(root.iterdir())[:20]}"
    )


def copy_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for p in src_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            # avoid name collisions by prefixing with parent folder name if needed
            target = dst_dir / p.name
            if target.exists():
                target = dst_dir / f"{p.parent.name}_{p.name}"
            shutil.copy2(p, target)
            count += 1
    return count


def main():
    dataset = "muhammadrehan00/chest-xray-dataset"

    downloaded_root = Path(kagglehub.dataset_download(dataset))
    print("Downloaded to:", downloaded_root)

    raw_root = Path("data/raw")
    # clean raw folder (optional). Comment out if you prefer incremental.
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)

    class_folders = find_class_folders(downloaded_root)
    print("Detected class folders:")
    for k, v in class_folders.items():
        print(f"  {k}: {v}")

    total = 0
    for cls, src in class_folders.items():
        dst = raw_root / cls
        n = copy_images(src, dst)
        total += n
        print(f"Copied {n} images to {dst}")

    print(f"✅ Done. Total images copied into {raw_root}: {total}")


if __name__ == "__main__":
    main()

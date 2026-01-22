import os, random, shutil
from pathlib import Path

random.seed(42)

RAW = Path("data/raw")
OUT = Path("data")
splits = {"train": 0.8, "val": 0.1, "test": 0.1}

def main():
    assert RAW.exists(), "Create data/raw/<class_name> with images first."
    for split in splits:
        (OUT / split).mkdir(parents=True, exist_ok=True)

    for class_dir in RAW.iterdir():
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.iterdir() if p.is_file()]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])

        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split, items in split_map.items():
            target = OUT / split / class_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for img in items:
                shutil.copy2(img, target / img.name)

    print("Done. Dataset split into data/train, data/val, data/test.")

if __name__ == "__main__":
    main()

import random
import shutil
from pathlib import Path

# Set paths
DATA_DIR = Path("data")
SOURCE_DIR = DATA_DIR / "PlantVillage"   # your extracted dataset folder

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def make_dirs(classes):
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in classes:
            (split / cls).mkdir(parents=True, exist_ok=True)

def split_class(cls_path, cls_name):
    images = list(cls_path.glob("*.*"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train+n_val]
    test_imgs = images[n_train+n_val:]

    for img in train_imgs:
        shutil.copy2(img, TRAIN_DIR / cls_name / img.name)

    for img in val_imgs:
        shutil.copy2(img, VAL_DIR / cls_name / img.name)

    for img in test_imgs:
        shutil.copy2(img, TEST_DIR / cls_name / img.name)

def main():
    classes = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes:", classes)

    make_dirs(classes)

    for cls in classes:
        print("Splitting:", cls)
        split_class(SOURCE_DIR / cls, cls)

    print("Done! Train/Val/Test created.")

if __name__ == "__main__":
    main()

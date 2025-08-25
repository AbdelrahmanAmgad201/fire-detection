import os
import glob
from collections import Counter

# Path to dataset root
DATASET_DIR = "home-fire-dataset"
splits = ["train", "val", "test"]

for split in splits:
    class_counter = Counter()
    total_objects = 0
    multi_object_images = 0

    label_dir = os.path.join(DATASET_DIR, split, "labels")
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    for lbl in label_files:
        with open(lbl, "r") as f:
            lines = [line.strip().split() for line in f.readlines() if line.strip()]
            if len(lines) > 1:   # more than 1 object in the image
                multi_object_images += 1
            for parts in lines:
                cls_id = int(parts[0])
                class_counter[cls_id] += 1
                total_objects += 1

    print(f"\n📂 {split.upper()} split:")
    print(f" Images: {len(label_files)}")
    print(f" Objects: {total_objects}")
    print(f" Images with >1 annotation: {multi_object_images}")
    for cls_id, count in sorted(class_counter.items()):
        print(f"   Class {cls_id}: {count}")

import os
import shutil
import random

base_dir = "wheat-disease-dataset-small"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

classes = ["BrownRust", "Healthy", "Mildew", "Septoria", "YellowRust"]

for cls in classes:
    files = os.listdir(os.path.join(train_dir, cls))
    random.shuffle(files)

    total = len(files)
    val_count = int(0.15 * total)
    test_count = int(0.15 * total)

    for f in files[:val_count]:
        shutil.move(
            os.path.join(train_dir, cls, f),
            os.path.join(val_dir, cls, f)
        )

    for f in files[val_count:val_count + test_count]:
        shutil.move(
            os.path.join(train_dir, cls, f),
            os.path.join(test_dir, cls, f)
        )

print("✅ Dataset split completed successfully")
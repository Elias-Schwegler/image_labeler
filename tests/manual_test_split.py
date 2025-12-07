import os
import json
import shutil
from src.splitter import split_dataset, organize_dataset
from src.data_loader import get_image_files

# Paths
# Paths
input_dir = os.path.abspath("data/raw/TEST_IMAGES")
output_dir = os.path.abspath("data/split_test_output")
labels_path = os.path.join(output_dir, "labels.json")

# Ensure directories
os.makedirs(output_dir, exist_ok=True)

# Generate dummy labels if not exist, based on actual images
image_files = get_image_files(input_dir)
if not image_files:
    print(f"No images found in {input_dir}. Please ensure TEST_IMAGES has content.")
    exit(1)

labeled_data = []
for img_path in image_files:
    labeled_data.append({
        "filename": os.path.basename(img_path),
        "original_path": img_path,
        "label": "dummy_label",
        "description": "auto-generated for testing"
    })

print(f"Generated {len(labeled_data)} dummy labels for testing.")

# Mocking the split process as done in app.py
# We need to find the files first. 
# In app.py, it tries to find files from labels.json or input_dir.
files = []
# Try to load from labels.json first (logic from app.py)
for item in labeled_data:
    if "original_path" in item and os.path.exists(item["original_path"]):
        files.append(item["original_path"])
    elif "filename" in item:
        potential_path = os.path.join(input_dir, item["filename"])
        if os.path.exists(potential_path):
            files.append(potential_path)

print(f"Found {len(files)} files to split.")

if not files:
    print("No files found. Cannot proceed.")
    exit()

# Split
train_files, test_files = split_dataset(files, 0.8)
print(f"Split into {len(train_files)} train and {len(test_files)} test.")

# Organize
temp_output_dir = os.path.abspath("data/split_test_output/organized")
if os.path.exists(temp_output_dir):
    shutil.rmtree(temp_output_dir)

print("Organizing dataset...")
train_dir, test_dir = organize_dataset(
    train_files, test_files, temp_output_dir, labeled_data=labeled_data
)

# Check results
train_labels_path = os.path.join(train_dir, "labels.json")
test_labels_path = os.path.join(test_dir, "labels.json")

print(f"Train labels exist: {os.path.exists(train_labels_path)}")
if os.path.exists(train_labels_path):
    with open(train_labels_path, "r") as f:
        print(f"Train labels count: {len(json.load(f))}")

print(f"Test labels exist: {os.path.exists(test_labels_path)}")
if os.path.exists(test_labels_path):
    with open(test_labels_path, "r") as f:
        print(f"Test labels count: {len(json.load(f))}")

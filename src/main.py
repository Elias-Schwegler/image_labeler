import argparse
import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm
from .data_loader import get_image_files, save_labels, ensure_directory
from .labeler import label_image
from .splitter import split_dataset, organize_dataset


def main():
    parser = argparse.ArgumentParser(description="Image Labeler CLI")
    parser.add_argument(
        "--path", type=str, help="Path to the directory containing images"
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/Test split ratio (default: 0.8)",
    )
    parser.add_argument("--ui", action="store_true", help="Start the Streamlit UI")
    parser.add_argument(
        "--output", type=str, help="Output directory for processed data"
    )

    args = parser.parse_args()

    if args.ui:
        print("Starting Streamlit UI...")
        # Get the path to app.py relative to this script
        app_path = Path(__file__).parent / "app.py"
        subprocess.run(["streamlit", "run", str(app_path)])
        return

    if not args.path:
        print("Error: --path is required unless --ui is specified.")
        return

    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Error: Input path '{args.path}' does not exist.")
        return

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path.cwd() / "output"

    ensure_directory(str(output_dir))
    print(f"Output directory: {output_dir}")

    # 1. Load Images
    print("Loading images...")
    image_files = get_image_files(str(input_path))
    print(f"Found {len(image_files)} images.")

    if not image_files:
        print("No images found.")
        return

    # 2. Label Images
    print("Labeling images (this may take a while)...")
    labeled_data = []
    for img_path in tqdm(image_files):
        label_result = label_image(img_path)
        # Add filename for reference
        label_result["filename"] = os.path.basename(img_path)
        label_result["original_path"] = img_path
        labeled_data.append(label_result)

    # Save labels
    labels_file = output_dir / "labels.json"
    save_labels(labeled_data, str(labels_file))
    print(f"Labels saved to {labels_file}")

    # 3. Split Dataset
    print(f"Splitting dataset with ratio {args.split_ratio}...")
    train_files, test_files = split_dataset(image_files, args.split_ratio)

    train_dir, test_dir = organize_dataset(train_files, test_files, str(output_dir))
    print(f"Dataset organized:")
    print(f"  Train: {len(train_files)} images in {train_dir}")
    print(f"  Test: {len(test_files)} images in {test_dir}")


if __name__ == "__main__":
    main()

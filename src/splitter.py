import os
import shutil
import random
from typing import List, Tuple, Dict, Any, Optional
import json
from .data_loader import ensure_directory


def split_dataset(
    image_files: List[str], split_ratio: float
) -> Tuple[List[str], List[str]]:
    """
    Split a list of image files into train and test sets based on the ratio.

    Args:
        image_files (List[str]): List of absolute paths to image files.
        split_ratio (float): The proportion of images to include in the training set.
                             Must be between 0 and 1 (e.g., 0.8 for 80% train).

    Returns:
        Tuple[List[str], List[str]]: A tuple containing (train_files, test_files).
    """
    if not 0 <= split_ratio <= 1:
        raise ValueError("Split ratio must be between 0 and 1")

    # Shuffle to ensure random split
    files_copy = image_files.copy()
    random.shuffle(files_copy)

    # Calculate split index
    split_index = int(len(files_copy) * split_ratio)
    train_files = files_copy[:split_index]
    test_files = files_copy[split_index:]

    return train_files, test_files


def organize_dataset(
    train_files: List[str],
    test_files: List[str],
    output_dir: str,
    labeled_data: Optional[List[Dict[str, Any]]] = None,
):
    """
    Copy files into train and test subdirectories in the output directory.
    If labeled_data is provided, it also splits the labels into train/labels.json and test/labels.json.

    Args:
        train_files (List[str]): List of paths for the training set.
        test_files (List[str]): List of paths for the test set.
        output_dir (str): The base directory where 'train' and 'test' folders will be created.
        labeled_data (Optional[List[Dict[str, Any]]]): List of label dictionaries.

    Returns:
        Tuple[str, str]: Paths to the created train and test directories.
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    ensure_directory(train_dir)
    ensure_directory(test_dir)

    # Copy files to destination
    for file_path in train_files:
        shutil.copy2(file_path, train_dir)

    for file_path in test_files:
        shutil.copy2(file_path, test_dir)

    # Split labels if provided
    if labeled_data:
        train_labels = []
        test_labels = []

        # Create sets for faster lookup
        # We use os.path.normpath/abspath to ensure consistency,
        # but simple string matching might suffice if paths are consistent.
        # Using basename might be safer if paths change, but duplicates are possible.
        # Let's try to match by original_path if available, or filename.

        train_files_set = set(os.path.abspath(f) for f in train_files)
        test_files_set = set(os.path.abspath(f) for f in test_files)

        for item in labeled_data:
            # Try to match by original_path
            original_path = item.get("original_path")
            if original_path:
                abs_path = os.path.abspath(original_path)
                if abs_path in train_files_set:
                    train_labels.append(item)
                elif abs_path in test_files_set:
                    test_labels.append(item)
            else:
                # Fallback: check if filename is in the list of basenames
                # This is less robust but might be necessary
                filename = item.get("filename")
                if filename:
                    if any(
                        os.path.basename(f) == filename for f in train_files
                    ):
                        train_labels.append(item)
                    elif any(
                        os.path.basename(f) == filename for f in test_files
                    ):
                        test_labels.append(item)

        # Save split labels
        with open(
            os.path.join(train_dir, "labels.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(train_labels, f, indent=4)

        with open(
            os.path.join(test_dir, "labels.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(test_labels, f, indent=4)

    return train_dir, test_dir

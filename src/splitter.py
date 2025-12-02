import os
import shutil
import random
from typing import List, Tuple
from pathlib import Path
from .data_loader import ensure_directory


def split_dataset(
    image_files: List[str], split_ratio: float
) -> Tuple[List[str], List[str]]:
    """
    Split a list of image files into train and test sets based on the ratio.
    ratio: Float between 0 and 1 (e.g., 0.8 for 80% train).
    """
    if not 0 <= split_ratio <= 1:
        raise ValueError("Split ratio must be between 0 and 1")

    # Shuffle to ensure random split
    files_copy = image_files.copy()
    random.shuffle(files_copy)

    split_index = int(len(files_copy) * split_ratio)
    train_files = files_copy[:split_index]
    test_files = files_copy[split_index:]

    return train_files, test_files


def organize_dataset(train_files: List[str], test_files: List[str], output_dir: str):
    """
    Copy files into train and test subdirectories in the output directory.
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    ensure_directory(train_dir)
    ensure_directory(test_dir)

    for file_path in train_files:
        shutil.copy2(file_path, train_dir)

    for file_path in test_files:
        shutil.copy2(file_path, test_dir)

    return train_dir, test_dir

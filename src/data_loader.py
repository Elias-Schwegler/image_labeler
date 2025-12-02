import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any


def get_image_files(directory: str) -> List[str]:
    """
    Get a list of image files in the specified directory.
    Supports common image extensions.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = []

    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for file in path.iterdir():
        if file.suffix.lower() in image_extensions:
            image_files.append(str(file.absolute()))

    return image_files


def save_labels(data: List[Dict[str, Any]], output_file: str):
    """
    Save labeled data to a JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_labels(input_file: str) -> List[Dict[str, Any]]:
    """
    Load labeled data from a JSON file.
    """
    if not os.path.exists(input_file):
        return []

    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_directory(directory: str):
    """
    Ensure a directory exists.
    """
    os.makedirs(directory, exist_ok=True)

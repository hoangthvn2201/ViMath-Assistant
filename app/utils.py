# app/utils.py

import os
import logging
from typing import Tuple, List, Dict, Any
from PIL import Image
import uuid
import datetime
import json
import numpy as np

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_temp_image(image: Image.Image, folder: str = "temp") -> str:
    """
    Save a PIL image to a temporary location for OCR or encoding.

    Args:
        image (PIL.Image): Image object.
        folder (str): Temporary directory path.

    Returns:
        str: Full path to saved image file.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(folder, filename)
    image.save(path)
    logger.info(f"Temporary image saved to {path}")
    return path


def get_current_timestamp() -> str:
    """
    Get current timestamp string (e.g., for logging or file naming).

    Returns:
        str: Timestamp string in ISO format.
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def format_score(score: float, digits: int = 3) -> str:
    """
    Format a float score to a readable string.

    Args:
        score (float): Raw score (e.g., similarity).
        digits (int): Decimal places to keep.

    Returns:
        str: Formatted score string.
    """
    return f"{score:.{digits}f}"


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove unwanted characters from OCR/text input.

    Args:
        text (str): Raw extracted or user input text.

    Returns:
        str: Cleaned text string.
    """
    return " ".join(text.strip().replace("\n", " ").split())


def safe_remove(path: str):
    """
    Remove a file if it exists, with exception handling.

    Args:
        path (str): Path to file.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed temp file: {path}")
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a .jsonl (JSON Lines) file and return a list of dicts.
    Each line is expected to be a valid JSON object.
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue  # optionally log malformed lines
    return data


def save_jsonl(path: str, data: List[Dict[str, Any]]):
    """
    Save a list of dicts to a .jsonl (JSON Lines) file.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def save_numpy_array(path: str, array: np.ndarray):
    """
    Save a numpy array to disk.
    """
    np.save(path, array)


def load_numpy_array(path: str) -> np.ndarray:
    """
    Load a numpy array from disk.
    """
    return np.load(path)


def ensure_dir(path: str):
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
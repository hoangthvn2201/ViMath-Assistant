# app/ocr.py

import os
import logging
from typing import List, Tuple

from paddleocr import PaddleOCR
from PIL import Image

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize OCR engine once (reuse across calls)
ocr_model = PaddleOCR(use_angle_cls=True, lang='vi', use_gpu=False)

def run_ocr(image_path: str) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Run OCR on a given image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Tuple:
            - str: Combined text from the OCR output.
            - List[Tuple[str, float]]: List of (text, confidence) tuples.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        result = ocr_model.ocr(image_path, cls=True)
        full_text = []
        text_with_conf = []

        # OCR returns a nested list of boxes
        for line in result[0]:
            text, confidence = line[1][0], line[1][1]
            full_text.append(text)
            text_with_conf.append((text, confidence))

        combined_text = " ".join(full_text)
        return combined_text, text_with_conf

    except Exception as e:
        logger.error(f"OCR failed on {image_path}: {str(e)}")
        raise RuntimeError("OCR processing error.") from e

def run_ocr_from_pil(image: Image.Image) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Run OCR on a PIL image directly (used when image is uploaded in-memory).

    Args:
        image (PIL.Image): The image to process.

    Returns:
        Tuple:
            - str: Combined extracted text.
            - List[Tuple[str, float]]: List of (text, confidence) tuples.
    """
    try:
        temp_path = "temp_ocr_img.png"
        image.save(temp_path)
        output = run_ocr(temp_path)
        os.remove(temp_path)
        return output
    except Exception as e:
        logger.error(f"OCR failed on PIL image: {str(e)}")
        raise RuntimeError("OCR from PIL failed.") from e
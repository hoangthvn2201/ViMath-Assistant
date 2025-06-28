# scripts/setup_vectorstore.py

import os
import json
from tqdm import tqdm
from PIL import Image

from app.embeddings.text_encoder import TextEncoder
from app.embeddings.image_encoder import ImageEncoder
from app.retriever import Retriever
from app.utils import clean_text

DATASET_PATH = "data/math_samples.json"     # JSONL or JSON list format
IMAGE_FOLDER = "data/images/"               # Where associated images (if any) are stored

# Initialize models
text_encoder = TextEncoder()
image_encoder = ImageEncoder()
retriever = Retriever(text_encoder=text_encoder, image_encoder=image_encoder)

def load_dataset(dataset_path: str):
    """
    Load dataset with math problems and optional images.

    Returns:
        List[Dict]: List of problems with keys: 'id', 'question', 'solution', (optional) 'image_filename'
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_vector_index():
    """
    Process dataset into vector store using the retriever.
    """
    problems = load_dataset(DATASET_PATH)
    added = 0

    for item in tqdm(problems, desc="Indexing problems"):
        problem_id = item["id"]
        question = clean_text(item["question"])
        solution = clean_text(item.get("solution", ""))
        image_path = None

        if "image_filename" in item:
            image_path = os.path.join(IMAGE_FOLDER, item["image_filename"])
            if not os.path.exists(image_path):
                print(f"[Warning] Image {image_path} not found, skipping image encoding.")

        image = None
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"[Error] Failed to open image {image_path}: {e}")

        retriever.add_to_index(
            problem_id=problem_id,
            text=question,
            solution=solution,
            image=image
        )
        added += 1

    print(f"✅ Indexed {added} math problems.")

if __name__ == "__main__":
    build_vector_index()
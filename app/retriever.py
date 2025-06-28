# app/retriever.py

from typing import List, Tuple
import numpy as np
import faiss

from embeddings.text_encoder import TextEncoder
from embeddings.image_encoder import ImageEncoder
from utils import load_jsonl
from PIL import Image

class Retriever:
    def __init__(
        self,
        index_path: str,
        db_path: str,
        text_encoder: TextEncoder,
        image_encoder: ImageEncoder = None,
        top_k: int = 5,
    ):
        self.index_path = index_path
        self.db_path = db_path
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.top_k = top_k

        self.examples = load_jsonl(self.db_path)
        self.index = faiss.read_index(self.index_path)

    def _encode_query(self, text: str, image: Image.Image = None) -> np.ndarray:
        """Generate embedding for a text (and optional image)."""
        if image is not None and self.image_encoder:
            image_vec = self.image_encoder.encode(image)
            text_vec = self.text_encoder.encode(text)
            combined = np.concatenate([text_vec, image_vec])
            return combined.astype("float32")
        else:
            return self.text_encoder.encode(text).astype("float32")

    def retrieve(self, text_query: str, image: np.ndarray = None) -> List[str]:
        """
        Retrieve top-k similar math examples from database given a text query and optional image.
        Returns list of example strings.
        """
        query_vec = self._encode_query(text_query, image)
        distances, indices = self.index.search(np.array([query_vec]), self.top_k)

        retrieved = []
        for idx in indices[0]:
            if 0 <= idx < len(self.examples):
                retrieved.append(self.examples[idx].get("content", ""))
        return retrieved

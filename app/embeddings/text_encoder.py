# app/embeddings/text_encoder.py
import os
from typing import List, Union
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEncoder:
    """
    Wrapper for Vietnamese Sentence-BERT embedding model.
    """
    def __init__(self, model_name_or_path: str = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base", device: str ='cpu'):
        """
        Load a pretrained Vietnamese SBERT model.

        Args:
            model_name_or_path (str): HuggingFace or local path to model.
        """
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = SentenceTransformer(self.model_name_or_path, device=self.device)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Encode one or more texts into embedding vectors.

        Args:
            texts (str or List[str]): Input string(s)

        Returns:
            np.ndarray: Embedding array with shape (n, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        return embeddings
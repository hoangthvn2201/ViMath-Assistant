# app/embeddings/image_encoder.py

import torch
from PIL import Image
from typing import Union, List
import numpy as np
from torchvision import transforms
# from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer

class ImageEncoder:
    """
    Wrapper for encoding images using a pretrained CLIP model.
    """

    def __init__(self, model_name_or_path: str = "clip-ViT-B-32", device: str = 'cpu'):
        """
        Initialize the image encoder.

        Args:
            model_name (str): HuggingFace model ID or local path to CLIP model.
            device (str): "cuda" or "cpu". Automatically chosen if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name_or_path, device=self.device)

    def encode(
        self,
        images: Union[Image.Image, List[Image.Image]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode image(s) into embedding vectors.

        Args:
            images (PIL.Image or list): One or more PIL images.
            normalize (bool): Whether to normalize embeddings to unit vectors.

        Returns:
            np.ndarray: Embeddings of shape (n_images, embedding_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        with torch.no_grad():
            image_features = self.encode(images, convert_to_numpy=True, normalize_embeddings=True)

        return image_features
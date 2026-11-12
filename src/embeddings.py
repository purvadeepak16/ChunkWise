"""Model loading and normalized embedding helpers."""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2)
def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def model_limit(model) -> int:
    value = getattr(model, "max_seq_length", None)
    return int(value) if value and value < 100000 else 256


def encode(model, texts):
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    vectors = np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))
    return vectors.astype(np.float32)

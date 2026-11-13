from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def compute_user_vector(
    history_embeddings: np.ndarray,
    decay_lambda: float = 0.85,
    normalize: bool = True,
) -> np.ndarray:
    """Compute user representation vector with exponential decay over reading history.

    Formula from plan:
    u = sum_{i=1}^N (lambda^(N-i) * v_i) / sum_{i=1}^N (lambda^(N-i))
    """
    if history_embeddings is None or len(history_embeddings) == 0:
        return np.zeros((384,), dtype=np.float32)

    history_array = np.asarray(history_embeddings, dtype=np.float32)
    if history_array.ndim == 1:
        history_array = history_array.reshape(1, -1)

    n_items = len(history_array)
    dim = history_array.shape[1]

    if n_items == 1:
        u = history_array[0].copy()
    else:
        # Most recent item is at index n_items - 1
        powers = np.arange(n_items - 1, -1, -1)  # [n-1, n-2, ..., 0]
        weights = np.power(decay_lambda, powers)
        u = np.sum(history_array * weights[:, np.newaxis], axis=0) / np.sum(weights)

    if normalize:
        norm = np.linalg.norm(u)
        u = u / norm if norm > 1e-8 else np.zeros(dim, dtype=np.float32)

    return u


class NewsEmbedder:
    """Generates dense vector embeddings using SentenceTransformers (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self._model: Any = None

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                token=self.hf_token,
            )
        return self._model

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            single_item = True
            texts = [texts]
        else:
            single_item = False

        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )

        if single_item:
            return embeddings[0]
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        if self._model is not None:
            return int(self.model.get_sentence_embedding_dimension() or 384)
        if "large" in self.model_name.lower():
            return 1024
        return 384

    @staticmethod
    def save_embeddings(
        output_path: Path,
        news_ids: list[str],
        embeddings: np.ndarray,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            news_ids=np.array(news_ids, dtype=object),
            embeddings=embeddings.astype(np.float32),
        )

    @staticmethod
    def load_embeddings(input_path: Path) -> tuple[list[str], np.ndarray]:
        data = np.load(input_path, allow_pickle=True)
        news_ids = [str(x) for x in data["news_ids"]]
        embeddings = data["embeddings"].astype(np.float32)
        return news_ids, embeddings

"""String embedder."""

from functools import lru_cache

import numpy as np
from tqdm.auto import trange

from raglite._config import RAGLiteConfig


@lru_cache(maxsize=128)
def _embed_string_batch(
    string_batch: tuple[str], *, config: RAGLiteConfig | None = None
) -> np.ndarray:
    # Embed a batch of strings.
    config = config or RAGLiteConfig()
    if len(string_batch) == 0:
        embeddings = np.zeros((0, config.embedder.n_embd()))
    else:
        embeddings = np.asarray(config.embedder.embed(string_batch))
    # Normalise embeddings to unit norm.
    if config.embedder_normalize:
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Cast to the configured dtype after normalisation.
    embeddings = embeddings.astype(config.embedder_dtype)
    return embeddings


def embed_strings(strings: list[str], *, config: RAGLiteConfig | None = None) -> np.ndarray:
    """Embed a list of strings as a NumPy array of row vectors."""
    assert isinstance(strings, list), "Input must be a list of strings"
    config = config or RAGLiteConfig()
    bs = config.embedder_batch_size
    brange = (
        trange(0, len(strings), bs, desc="Embedding", unit="batch", dynamic_ncols=True)
        if len(strings) > bs
        else range(0, max(1, len(strings)), bs)
    )
    batches = [_embed_string_batch(tuple(strings[i : i + bs]), config=config) for i in brange]
    embeddings = np.vstack(batches)
    return embeddings

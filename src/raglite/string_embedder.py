"""String embedder."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
import numpy.typing as npt
from llama_cpp import Llama
from tqdm.auto import trange

from raglite.llm import default_embedder


@lru_cache(maxsize=128)
def _embed_string_batch(
    string_batch: tuple[str],
    dtype: npt.DTypeLike = np.float16,
    embedder: Callable[[], Llama] = default_embedder,
) -> np.ndarray:
    if len(string_batch) == 0:
        embeddings = np.zeros((0, embedder().n_embd()), dtype=dtype)
    else:
        embeddings = np.asarray(embedder().embed(string_batch), dtype=dtype)
    return embeddings


def embed_strings(
    strings: list[str], batch_size: int = 128, dtype: npt.DTypeLike = np.float16
) -> np.ndarray:
    """Embed a list of strings as a NumPy array of row vectors."""
    brange = (
        trange(0, len(strings), batch_size, desc="Embedding", unit="batch", dynamic_ncols=True)
        if len(strings) > batch_size
        else range(0, max(1, len(strings)), batch_size)
    )
    batches = [_embed_string_batch(tuple(strings[i : i + batch_size]), dtype=dtype) for i in brange]
    embeddings = np.vstack(batches)
    return embeddings

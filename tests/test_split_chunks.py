"""Test RAGLite's chunk splitting functionality."""

import numpy as np
import pytest

from raglite._split_chunks import split_chunks


@pytest.mark.parametrize(
    "chunklets",
    [
        pytest.param([], id="one_chunk:no_chunklets"),
        pytest.param(["Hello world"], id="one_chunk:one_chunklet"),
        pytest.param(["Hello world"] * 2, id="one_chunk:two_chunklets"),
        pytest.param(["Hello world"] * 3, id="one_chunk:three_chunklets"),
        pytest.param(["Hello world"] * 100, id="one_chunk:many_chunklets"),
        pytest.param(["Hello world", "X" * 1000], id="n_chunks:two_chunklets_a"),
        pytest.param(["X" * 1000, "Hello world"], id="n_chunks:two_chunklets_b"),
        pytest.param(["Hello world", "X" * 1000, "X" * 1000], id="n_chunks:three_chunklets_a"),
        pytest.param(["X" * 1000, "Hello world", "X" * 1000], id="n_chunks:three_chunklets_b"),
        pytest.param(["X" * 1000, "X" * 1000, "Hello world"], id="n_chunks:three_chunklets_c"),
        pytest.param(["X" * 1000] * 100, id="n_chunks:many_chunklets_a"),
        pytest.param(["X" * 100] * 1000, id="n_chunks:many_chunklets_b"),
    ],
)
def test_edge_cases(chunklets: list[str]) -> None:
    """Test chunk splitting edge cases."""
    chunklet_embeddings = np.ones((len(chunklets), 768)).astype(np.float16)
    chunks, chunk_embeddings = split_chunks(chunklets, chunklet_embeddings, max_size=1440)
    assert isinstance(chunks, list)
    assert isinstance(chunk_embeddings, list)
    assert len(chunk_embeddings) == (len(chunks) if chunklets else 1)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(isinstance(chunk_embedding, np.ndarray) for chunk_embedding in chunk_embeddings)
    assert all(ce.dtype == chunklet_embeddings.dtype for ce in chunk_embeddings)
    assert sum(ce.shape[0] for ce in chunk_embeddings) == chunklet_embeddings.shape[0]
    assert all(ce.shape[1] == chunklet_embeddings.shape[1] for ce in chunk_embeddings)


@pytest.mark.parametrize(
    "chunklets",
    [
        pytest.param(["Hello world" * 1000] + ["X"] * 100, id="first"),
        pytest.param(["X"] * 50 + ["Hello world" * 1000] + ["X"] * 50, id="middle"),
        pytest.param(["X"] * 100 + ["Hello world" * 1000], id="last"),
    ],
)
def test_long_chunklet(chunklets: list[str]) -> None:
    """Test chunking on chunklets that are too long."""
    chunklet_embeddings = np.ones((len(chunklets), 768)).astype(np.float16)
    with pytest.raises(ValueError, match="Chunklet larger than chunk max_size detected."):
        _ = split_chunks(chunklets, chunklet_embeddings, max_size=1440)

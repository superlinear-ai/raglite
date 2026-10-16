"""Test RAGLite's embedding functionality."""

from pathlib import Path

import numpy as np
import pytest

from raglite import RAGLiteConfig
from raglite._embed import embed_strings
from raglite._markdown import document_to_markdown
from raglite._split_sentences import split_sentences


def test_embed(embedder: str) -> None:
    """Test embedding a document."""
    raglite_test_config = RAGLiteConfig(embedder=embedder, embedder_normalize=True)
    doc_path = Path(__file__).parent / "specrel.pdf"  # Einstein's special relativity paper.
    doc = document_to_markdown(doc_path)
    sentences = split_sentences(doc, max_len=raglite_test_config.chunk_max_size)
    sentence_embeddings = embed_strings(sentences, config=raglite_test_config)
    assert isinstance(sentences, list)
    assert isinstance(sentence_embeddings, np.ndarray)
    assert len(sentences) == len(sentence_embeddings)
    assert sentence_embeddings.shape[1] >= 128  # noqa: PLR2004
    assert sentence_embeddings.dtype == np.float16
    assert np.all(np.isfinite(sentence_embeddings))
    assert np.allclose(np.linalg.norm(sentence_embeddings, axis=1), 1.0, rtol=1e-3)


def test_embed_raises_when_chunklet_exceeds_content_budget(embedder: str) -> None:
    """A chunklet whose token count exceeds the local content budget should raise, not hang.

    Regression test for https://github.com/superlinear-ai/raglite/issues/188: late chunking's
    segment-boundary search could return a zero-width segment when a single sentence needed more
    tokens than were available in the content budget at its position, causing an infinite loop in
    `embed_strings_with_late_chunking`'s segmentation `while` loop instead of a clear error.
    """
    if not embedder.startswith("llama-cpp-python"):
        return  # The bug is specific to the late chunking path used by llama-cpp-python embedders.
    raglite_test_config = RAGLiteConfig(embedder=embedder, embedder_normalize=True)
    # Build one sentence far larger than any embedder context window could allow as content, by
    # using many distinct short "words" so the tokenizer can't merge them into few tokens.
    oversized_sentence = " ".join(f"tok{i}" for i in range(4000))
    sentences = [oversized_sentence, "A short trailing sentence."]
    with pytest.raises(ValueError, match="exceeds the remaining segment content budget"):
        embed_strings(sentences, config=raglite_test_config)

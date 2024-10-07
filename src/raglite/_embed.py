"""String embedder."""

from functools import partial
from typing import Literal

import numpy as np
from litellm import embedding
from llama_cpp import LLAMA_POOLING_TYPE_NONE, Llama
from tqdm.auto import tqdm, trange

from raglite._config import RAGLiteConfig
from raglite._litellm import LlamaCppPythonLLM
from raglite._typing import FloatMatrix, IntVector


def _embed_sentences_with_late_chunking(  # noqa: PLR0915
    sentences: list[str], *, config: RAGLiteConfig | None = None
) -> FloatMatrix:
    """Embed a document's sentences with late chunking."""

    def _count_tokens(
        sentences: list[str], embedder: Llama, sentinel_char: str, sentinel_tokens: list[int]
    ) -> list[int]:
        # Join the sentences with the sentinel token and tokenise the result.
        sentences_tokens = np.asarray(
            embedder.tokenize(sentinel_char.join(sentences).encode(), add_bos=False), dtype=np.intp
        )
        # Map all sentinel token variants to the first one.
        for sentinel_token in sentinel_tokens[1:]:
            sentences_tokens[sentences_tokens == sentinel_token] = sentinel_tokens[0]
        # Count how many tokens there are in between sentinel tokens to recover the token counts.
        sentinel_indices = np.where(sentences_tokens == sentinel_tokens[0])[0]
        num_tokens = np.diff(sentinel_indices, prepend=0, append=len(sentences_tokens))
        assert len(num_tokens) == len(sentences), f"Sentinel `{sentinel_char}` appears in document"
        num_tokens_list: list[int] = num_tokens.tolist()
        return num_tokens_list

    def _create_segment(
        content_start_index: int,
        max_tokens_preamble: int,
        max_tokens_content: int,
        num_tokens: IntVector,
    ) -> tuple[int, int]:
        # Compute the segment sentence start index so that the segment preamble has no more than
        # max_tokens_preamble tokens between [segment_start_index, content_start_index).
        cumsum_backwards = np.cumsum(num_tokens[:content_start_index][::-1])
        offset_preamble = np.searchsorted(cumsum_backwards, max_tokens_preamble, side="right")
        segment_start_index = content_start_index - int(offset_preamble)
        # Allow a larger segment content if we didn't use all of the allowed preamble tokens.
        max_tokens_content = max_tokens_content + (
            max_tokens_preamble - np.sum(num_tokens[segment_start_index:content_start_index])
        )
        # Compute the segment sentence end index so that the segment content has no more than
        # max_tokens_content tokens between [content_start_index, segment_end_index).
        cumsum_forwards = np.cumsum(num_tokens[content_start_index:])
        offset_segment = np.searchsorted(cumsum_forwards, max_tokens_content, side="right")
        segment_end_index = content_start_index + int(offset_segment)
        return segment_start_index, segment_end_index

    # Assert that we're using a llama-cpp-python model, since API-based embedding models don't
    # support outputting token-level embeddings.
    config = config or RAGLiteConfig()
    assert config.embedder.startswith("llama-cpp-python")
    embedder = LlamaCppPythonLLM.llm(
        config.embedder, embedding=True, pooling_type=LLAMA_POOLING_TYPE_NONE
    )
    n_ctx = embedder.n_ctx()
    n_batch = embedder.n_batch
    # Identify the tokens corresponding to a sentinel character.
    sentinel_char = "⊕"
    sentinel_test = f"A{sentinel_char}B {sentinel_char} C.\n{sentinel_char}D"
    sentinel_tokens = [
        token
        for token in embedder.tokenize(sentinel_test.encode(), add_bos=False)
        if sentinel_char in embedder.detokenize([token]).decode()
    ]
    assert len(sentinel_tokens), f"Sentinel `{sentinel_char}` not supported by embedder"
    # Compute the number of tokens per sentence. We use a method based on a sentinel token to
    # minimise the number of calls to embedder.tokenize, which incurs a significant overhead
    # (presumably to load the tokenizer) [1].
    # TODO: Make token counting faster and more robust once [1] is fixed.
    # [1] https://github.com/abetlen/llama-cpp-python/issues/1763
    num_tokens_list: list[int] = []
    sentence_batch, sentence_batch_len = [], 0
    for i, sentence in enumerate(sentences):
        sentence_batch.append(sentence)
        sentence_batch_len += len(sentence)
        if i == len(sentences) - 1 or sentence_batch_len > (n_ctx // 2):
            num_tokens_list.extend(
                _count_tokens(sentence_batch, embedder, sentinel_char, sentinel_tokens)
            )
            sentence_batch, sentence_batch_len = [], 0
    num_tokens = np.asarray(num_tokens_list, dtype=np.intp)
    # Compute the maximum number of tokens for each segment's preamble and content.
    # Unfortunately, llama-cpp-python truncates the input to n_batch tokens and crashes if you try
    # to increase it [1]. Until this is fixed, we have to limit max_tokens to n_batch.
    # TODO: Improve the context window size once [1] is fixed.
    # [1] https://github.com/abetlen/llama-cpp-python/issues/1762
    max_tokens = min(n_ctx, n_batch) - 16
    max_tokens_preamble = round(0.382 * max_tokens)  # Golden ratio.
    max_tokens_content = max_tokens - max_tokens_preamble
    # Compute a list of segments, each consisting of a preamble and content.
    segments = []
    content_start_index = 0
    while content_start_index < len(sentences):
        segment_start_index, segment_end_index = _create_segment(
            content_start_index, max_tokens_preamble, max_tokens_content, num_tokens
        )
        segments.append((segment_start_index, content_start_index, segment_end_index))
        content_start_index = segment_end_index
    # Embed the segments and apply late chunking.
    sentence_embeddings_list: list[FloatMatrix] = []
    if len(segments) > 1 or segments[0][2] > 128:  # noqa: PLR2004
        segments = tqdm(segments, desc="Embedding", unit="segment", dynamic_ncols=True)
    for segment in segments:
        # Get the token embeddings of the entire segment, including preamble and content.
        segment_start_index, content_start_index, segment_end_index = segment
        segment_sentences = sentences[segment_start_index:segment_end_index]
        segment_embedding = np.asarray(embedder.embed("".join(segment_sentences)))
        # Split the segment embeddings into embedding matrices per sentence.
        segment_tokens = num_tokens[segment_start_index:segment_end_index]
        sentence_size = np.round(
            len(segment_embedding) * (segment_tokens / np.sum(segment_tokens))
        ).astype(np.intp)
        sentence_matrices = np.split(segment_embedding, np.cumsum(sentence_size)[:-1])
        # Compute the segment sentence embeddings by averaging the token embeddings.
        content_sentence_embeddings = [
            np.mean(sentence_matrix, axis=0, keepdims=True)
            for sentence_matrix in sentence_matrices[content_start_index - segment_start_index :]
        ]
        sentence_embeddings_list.append(np.vstack(content_sentence_embeddings))
    sentence_embeddings = np.vstack(sentence_embeddings_list)
    # Normalise the sentence embeddings to unit norm and cast to half precision.
    if config.embedder_normalize:
        sentence_embeddings /= np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
    sentence_embeddings = sentence_embeddings.astype(np.float16)
    return sentence_embeddings


def _embed_sentences_with_windowing(
    sentences: list[str], *, config: RAGLiteConfig | None = None
) -> FloatMatrix:
    """Embed a document's sentences with windowing."""

    def _embed_string_batch(string_batch: list[str], *, config: RAGLiteConfig) -> FloatMatrix:
        # Embed the batch of strings.
        if config.embedder.startswith("llama-cpp-python"):
            # LiteLLM doesn't yet support registering a custom embedder, so we handle it here.
            # Additionally, we explicitly manually pool the token embeddings to obtain sentence
            # embeddings because token embeddings are universally supported, while sequence
            # embeddings are only supported by some models.
            embedder = LlamaCppPythonLLM.llm(
                config.embedder, embedding=True, pooling_type=LLAMA_POOLING_TYPE_NONE
            )
            embeddings = np.asarray([np.mean(row, axis=0) for row in embedder.embed(string_batch)])
        else:
            # Use LiteLLM's API to embed the batch of strings.
            response = embedding(config.embedder, string_batch)
            embeddings = np.asarray([item["embedding"] for item in response["data"]])
        # Normalise the embeddings to unit norm and cast to half precision.
        if config.embedder_normalize:
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings.astype(np.float16)
        return embeddings

    # Window the sentences with a lookback of `config.embedder_sentence_window_size - 1` sentences.
    config = config or RAGLiteConfig()
    sentence_windows = [
        "".join(sentences[max(0, i - (config.embedder_sentence_window_size - 1)) : i + 1])
        for i in range(len(sentences))
    ]
    # Embed the sentence windows in batches.
    batch_size = 64
    batch_range = (
        partial(trange, desc="Embedding", unit="batch", dynamic_ncols=True)
        if len(sentence_windows) > batch_size
        else range
    )
    batch_embeddings = [
        _embed_string_batch(sentence_windows[i : i + batch_size], config=config)
        for i in batch_range(0, len(sentence_windows), batch_size)  # type: ignore[operator]
    ]
    sentence_embeddings = np.vstack(batch_embeddings)
    return sentence_embeddings


def sentence_embedding_type(
    *,
    config: RAGLiteConfig | None = None,
) -> Literal["late_chunking", "windowing"]:
    """Return the type of sentence embeddings."""
    config = config or RAGLiteConfig()
    return "late_chunking" if config.embedder.startswith("llama-cpp-python") else "windowing"


def embed_sentences(sentences: list[str], *, config: RAGLiteConfig | None = None) -> FloatMatrix:
    """Embed the sentences of a document as a NumPy matrix with one row per sentence."""
    config = config or RAGLiteConfig()
    if sentence_embedding_type(config=config) == "late_chunking":
        sentence_embeddings = _embed_sentences_with_late_chunking(sentences, config=config)
    else:
        sentence_embeddings = _embed_sentences_with_windowing(sentences, config=config)
    return sentence_embeddings

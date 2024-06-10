"""Large Language Models."""

from functools import lru_cache

from llama_cpp import Llama


@lru_cache(maxsize=1)
def default_llm() -> Llama:
    """Get default LLM."""
    llm = Llama.from_pretrained(
        repo_id="second-state/Yi-1.5-9B-Chat-16K-GGUF",
        filename="*Q4_K_M.gguf",
        n_ctx=0,
        n_gpu_layers=-1,
        use_mlock=True,
        use_mmap=False,
        verbose=False,
    )
    return llm


@lru_cache(maxsize=1)
def default_embedder() -> Llama:
    """Get default embedder."""
    embedder = Llama.from_pretrained(
        # https://github.com/Snowflake-Labs/arctic-embed
        repo_id="ChristianAzinn/snowflake-arctic-embed-l-gguf",
        filename="*f16.GGUF",
        n_ctx=0,
        n_gpu_layers=-1,
        use_mlock=True,
        use_mmap=False,
        verbose=False,
        embedding=True,
    )
    return embedder

"""验证没有语言特征的查询或文档安全回退到通用重排器."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from raglite import RAGLiteConfig, rerank_chunks
from raglite._database import Chunk


@pytest.mark.parametrize(
    ("query", "body"),
    [
        ("123", "123456"),
        ("123", "The invoice is due next month."),
        ("What is the total?", "123456"),
    ],
)
def test_rerank_language_falls_back(query: str, body: str) -> None:
    """验证语言检测失败时的路由边界."""
    generic = Mock()
    generic.rank.return_value = SimpleNamespace(results=[SimpleNamespace(doc_id=0)])
    english = Mock()
    config = RAGLiteConfig(
        llm="unused", embedder="unused", reranker={"en": english, "other": generic}
    )
    chunk = Chunk(id="chunk", document_id="doc", index=0, headings="", body=body)
    assert rerank_chunks(query, [chunk], config=config) == [chunk]
    generic.rank.assert_called_once_with(query=query, docs=[str(chunk)])
    english.rank.assert_not_called()


def test_unknown_language_without_generic_reranker_preserves_order() -> None:
    """验证语言检测失败时的路由边界."""
    english = Mock()
    config = RAGLiteConfig(llm="unused", embedder="unused", reranker={"en": english})
    chunk = Chunk(id="chunk", document_id="doc", index=0, headings="", body="123456")
    assert rerank_chunks("123", [chunk], config=config) == [chunk]
    english.rank.assert_not_called()


def test_single_reranker_does_not_require_language_detection() -> None:
    """验证语言检测失败时的路由边界."""
    generic = Mock()
    generic.rank.return_value = SimpleNamespace(results=[SimpleNamespace(doc_id=0)])
    config = RAGLiteConfig(llm="unused", embedder="unused", reranker=generic)
    chunk = Chunk(id="chunk", document_id="doc", index=0, headings="", body="123456")
    assert rerank_chunks("123", [chunk], config=config) == [chunk]
    generic.rank.assert_called_once()


def test_detectable_english_keeps_language_specific_reranker() -> None:
    """验证正常英语输入继续选择英语模型."""
    english = Mock()
    english.rank.return_value = SimpleNamespace(results=[SimpleNamespace(doc_id=0)])
    generic = Mock()
    config = RAGLiteConfig(
        llm="unused", embedder="unused", reranker={"en": english, "other": generic}
    )
    text = "The quick brown fox jumps over the lazy dog in the garden."
    chunk = Chunk(id="chunk", document_id="doc", index=0, headings="", body=text)
    assert rerank_chunks(text, [chunk], config=config) == [chunk]
    english.rank.assert_called_once()
    generic.rank.assert_not_called()


def test_empty_chunks_do_not_call_reranker() -> None:
    """验证空候选不会加载或调用重排流程."""
    generic = Mock()
    config = RAGLiteConfig(llm="unused", embedder="unused", reranker={"other": generic})
    assert rerank_chunks("123", [], config=config) == []
    generic.rank.assert_not_called()

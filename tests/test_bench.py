"""Test RAGLite's benchmarking functionality."""

from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import MagicMock

import pytest


class _Doc(NamedTuple):
    doc_id: str
    text: str


class _Dataset:
    """A minimal stand-in for an `ir_datasets` dataset."""

    def __init__(self, docs: list[_Doc]) -> None:
        self.docs = docs

    def docs_namespace(self) -> str:
        return "test"

    def docs_count(self) -> int:
        return len(self.docs)

    def docs_iter(self) -> list[_Doc]:
        return self.docs


@pytest.mark.usefixtures("cp1252_default_text_encoding")
def test_openai_vector_store_documents_are_uploaded_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that documents are written to disk as UTF-8 before they are uploaded."""
    bench = pytest.importorskip("raglite._bench")  # Requires the `bench` extra.
    monkeypatch.setattr(bench, "user_data_dir", lambda *_, **__: str(tmp_path))
    # Create an evaluator with a mocked OpenAI client that records the uploaded file contents.
    expected_text = "Maxwell’s “electrodynamics” — 電磁気学"  # noqa: RUF001
    dataset: Any = _Dataset([_Doc(doc_id="doc-1", text=expected_text)])
    evaluator = bench.OpenAIVectorStoreEvaluator(dataset)
    uploaded_contents: list[bytes] = []
    client = MagicMock()
    client.vector_stores.file_batches.upload_and_poll.side_effect = lambda **kwargs: (
        uploaded_contents.extend(file.read() for file in kwargs["files"])
    )
    evaluator.__dict__["client"] = client  # Override the cached_property.
    # Insert the documents.
    evaluator.insert_documents()
    assert uploaded_contents == [expected_text.encode("utf-8")]

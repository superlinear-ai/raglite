"""Benchmarking with TREC runs."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import Any

from ir_datasets.datasets.base import Dataset
from ir_measures import ScoredDoc, read_trec_run
from platformdirs import user_data_dir
from slugify import slugify
from tqdm.auto import tqdm

from raglite._config import RAGLiteConfig


class IREvaluator(ABC):
    def __init__(
        self,
        dataset: Dataset,
        *,
        num_results: int = 10,
        insert_variant: str | None = None,
        search_variant: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.num_results = num_results
        self.insert_variant = insert_variant
        self.search_variant = search_variant
        self.insert_id = (
            slugify(self.__class__.__name__.lower().replace("evaluator", ""))
            + (f"_{slugify(insert_variant)}" if insert_variant else "")
            + f"_{slugify(dataset.docs_namespace())}"
        )
        self.search_id = (
            self.insert_id
            + f"@{num_results}"
            + (f"_{slugify(search_variant)}" if search_variant else "")
        )
        self.cwd = Path(user_data_dir("raglite", ensure_exists=True))

    @abstractmethod
    def insert_documents(self, max_workers: int | None = None) -> None:
        """Insert all of the dataset's documents into the search index."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_id: str, query: str, *, num_results: int = 10) -> list[ScoredDoc]:
        """Search for documents given a query."""
        raise NotImplementedError

    @property
    def trec_run_filename(self) -> str:
        return f"{self.search_id}.trec"

    @property
    def trec_run_filepath(self) -> Path:
        return self.cwd / self.trec_run_filename

    def score(self) -> Generator[ScoredDoc, None, None]:
        """Read or compute a TREC run."""
        if self.trec_run_filepath.exists():
            yield from read_trec_run(self.trec_run_filepath.as_posix())  # type: ignore[no-untyped-call]
            return
        if not self.search("q0", next(self.dataset.queries_iter()).text):
            self.insert_documents()
        with self.trec_run_filepath.open(mode="w") as trec_run_file:
            for query in tqdm(
                self.dataset.queries_iter(),
                total=self.dataset.queries_count(),
                desc="Running queries",
                unit="query",
                dynamic_ncols=True,
            ):
                results = self.search(query.query_id, query.text, num_results=self.num_results)
                unique_results = {doc.doc_id: doc for doc in sorted(results, key=lambda d: d.score)}
                top_results = sorted(unique_results.values(), key=lambda d: d.score, reverse=True)
                top_results = top_results[: self.num_results]
                for rank, scored_doc in enumerate(top_results):
                    trec_line = f"{query.query_id} 0 {scored_doc.doc_id} {rank} {scored_doc.score} {self.trec_run_filename}\n"
                    trec_run_file.write(trec_line)
                    yield scored_doc


class RAGLiteEvaluator(IREvaluator):
    def __init__(
        self,
        dataset: Dataset,
        *,
        num_results: int = 10,
        insert_variant: str | None = None,
        search_variant: str | None = None,
        config: RAGLiteConfig | None = None,
    ):
        super().__init__(
            dataset,
            num_results=num_results,
            insert_variant=insert_variant,
            search_variant=search_variant,
        )
        self.db_filepath = self.cwd / f"{self.insert_id}.db"
        db_url = f"duckdb:///{self.db_filepath.as_posix()}"
        self.config = replace(config or RAGLiteConfig(), db_url=db_url)

    def insert_documents(self, max_workers: int | None = None) -> None:
        from raglite import Document, insert_documents

        documents = [
            Document.from_text(doc.text, id=doc.doc_id) for doc in self.dataset.docs_iter()
        ]
        insert_documents(documents, max_workers=max_workers, config=self.config)

    def update_query_adapter(self, num_evals: int = 1024) -> None:
        from raglite import insert_evals, update_query_adapter
        from raglite._database import IndexMetadata

        if (
            self.config.vector_search_query_adapter
            and IndexMetadata.get(config=self.config).get("query_adapter") is None
        ):
            insert_evals(num_evals=num_evals, config=self.config)
            update_query_adapter(config=self.config)

    def search(self, query_id: str, query: str, *, num_results: int = 10) -> list[ScoredDoc]:
        from raglite import retrieve_chunks, vector_search

        self.update_query_adapter()
        chunk_ids, scores = vector_search(query, num_results=2 * num_results, config=self.config)
        chunks = retrieve_chunks(chunk_ids, config=self.config)
        scored_docs = [
            ScoredDoc(query_id=query_id, doc_id=chunk.document.id, score=score)
            for chunk, score in zip(chunks, scores, strict=True)
        ]
        return scored_docs


class LlamaIndexEvaluator(IREvaluator):
    def __init__(
        self,
        dataset: Dataset,
        *,
        num_results: int = 10,
        insert_variant: str | None = None,
        search_variant: str | None = None,
    ):
        super().__init__(
            dataset,
            num_results=num_results,
            insert_variant=insert_variant,
            search_variant=search_variant,
        )
        self.embedder = "text-embedding-3-large"
        self.embedder_dim = 3072
        self.persist_path = self.cwd / self.insert_id

    def insert_documents(self, max_workers: int | None = None) -> None:
        # Adapted from https://docs.llamaindex.ai/en/stable/examples/vector_stores/FaissIndexDemo/.
        import faiss
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.faiss import FaissVectorStore

        self.persist_path.mkdir(parents=True, exist_ok=True)
        faiss_index = faiss.IndexHNSWFlat(self.embedder_dim, 32, faiss.METRIC_INNER_PRODUCT)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        index = VectorStoreIndex.from_documents(
            [
                Document(id_=doc.doc_id, text=doc.text, metadata={"filename": doc.doc_id})
                for doc in self.dataset.docs_iter()
            ],
            storage_context=StorageContext.from_defaults(vector_store=vector_store),
            embed_model=OpenAIEmbedding(model=self.embedder, dimensions=self.embedder_dim),
            show_progress=True,
        )
        index.storage_context.persist(persist_dir=self.persist_path)

    @cached_property
    def index(self) -> Any:
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.faiss import FaissVectorStore

        vector_store = FaissVectorStore.from_persist_dir(persist_dir=self.persist_path.as_posix())
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=self.persist_path.as_posix()
        )
        embed_model = OpenAIEmbedding(model=self.embedder, dimensions=self.embedder_dim)
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        return index

    def search(self, query_id: str, query: str, *, num_results: int = 10) -> list[ScoredDoc]:
        if not self.persist_path.exists():
            self.insert_documents()
        retriever = self.index.as_retriever(similarity_top_k=2 * num_results)
        nodes = retriever.retrieve(query)
        scored_docs = [
            ScoredDoc(
                query_id=query_id,
                doc_id=node.metadata.get("filename", node.id_),
                score=node.score if node.score is not None else 1.0,
            )
            for node in nodes
        ]
        return scored_docs


class OpenAIVectorStoreEvaluator(IREvaluator):
    def __init__(
        self,
        dataset: Dataset,
        *,
        num_results: int = 10,
        insert_variant: str | None = None,
        search_variant: str | None = None,
    ):
        super().__init__(
            dataset,
            num_results=num_results,
            insert_variant=insert_variant,
            search_variant=search_variant,
        )
        self.vector_store_name = dataset.docs_namespace() + (
            f"_{slugify(insert_variant)}" if insert_variant else ""
        )

    @cached_property
    def client(self) -> Any:
        import openai

        return openai.OpenAI()

    @property
    def vector_store_id(self) -> str | None:
        vector_stores = self.client.vector_stores.list()
        vector_store = next((vs for vs in vector_stores if vs.name == self.vector_store_name), None)
        if vector_store is None:
            return None
        if vector_store.file_counts.failed > 0:
            warnings.warn(
                f"Vector store {vector_store.name} has {vector_store.file_counts.failed} failed files.",
                stacklevel=2,
            )
        if vector_store.file_counts.in_progress > 0:
            error_message = f"Vector store {vector_store.name} has {vector_store.file_counts.in_progress} files in progress."
            raise RuntimeError(error_message)
        return vector_store.id  # type: ignore[no-any-return]

    def insert_documents(self, max_workers: int | None = None) -> None:
        import tempfile
        from pathlib import Path

        vector_store = self.client.vector_stores.create(name=self.vector_store_name)
        files, max_files_per_batch = [], 32
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, doc in tqdm(
                enumerate(self.dataset.docs_iter()),
                total=self.dataset.docs_count(),
                desc="Inserting documents",
                unit="document",
                dynamic_ncols=True,
            ):
                if not doc.text.strip():
                    continue
                temp_file = Path(temp_dir) / f"{slugify(doc.doc_id)}.txt"
                temp_file.write_text(doc.text)
                files.append(temp_file.open("rb"))
                if len(files) == max_files_per_batch or (i == self.dataset.docs_count() - 1):
                    self.client.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store.id, files=files, max_concurrency=max_workers
                    )
                    for f in files:
                        f.close()
                    files = []

    @cached_property
    def filename_to_doc_id(self) -> dict[str, str]:
        return {f"{slugify(doc.doc_id)}.txt": doc.doc_id for doc in self.dataset.docs_iter()}

    def search(self, query_id: str, query: str, *, num_results: int = 10) -> list[ScoredDoc]:
        if not self.vector_store_id:
            return []
        response = self.client.vector_stores.search(
            vector_store_id=self.vector_store_id, query=query, max_num_results=2 * num_results
        )
        scored_docs = [
            ScoredDoc(
                query_id=query_id,
                doc_id=self.filename_to_doc_id[result.filename],
                score=result.score,
            )
            for result in response
        ]
        return scored_docs

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Annotated
from enum import Enum
from pydantic import Field

from dotenv import load_dotenv
from rerankers import Reranker
from tqdm import tqdm

from crag.models.utils import extract_year_from_last_modified, html_to_md, read_jsonl
from raglite import RAGLiteConfig, add_context, hybrid_search, rag, vector_search
from raglite._database import ChunkSpan
from raglite._extract import expand_document_metadata

load_dotenv()

class _Domain(str, Enum):
    sports = "sports"
    music = "music"
    movie = "movie"
    popculture = "popculture"
    forums = "forums"
    travel_geography = "travel_geography"
    technology = "technology"
    health_nutrition = "health_nutrition"
    science_research = "science_research"
    other = "other"


class _ContentKind(str, Enum):
    profile_bio = "profile_bio"
    reference_explainer = "reference_explainer"
    news_article = "news_article"
    stats_scores = "stats_scores"
    comparison = "comparison"
    ranking_list = "ranking_list"
    review = "review"
    guide_howto = "guide_howto"
    forum_thread = "forum_thread"
    other = "other"


METADATA_FIELDS = {
    "domains": Annotated[
        list[_Domain] | None,
        Field(
            default_factory=list,
            max_length=3,
            description="Broad topic areas specifying the general subject matter of the page (max 3).",
        ),
    ],
    "primary_entity": Annotated[
        list[str] | None,
        Field(
            default_factory=list,
            max_length=2,
            description="Main subjects of the page in lowercase (max 2). An entity can be a person, place, thing, concept, etc. "
            "It should be specific enough to distinguish the page from others, but not so specific that it only applies to a single page. "
            "Example entities: 'lebron james', 'guitar pedals', 'hiking', 'national park'."),
    ],
    "content_type": Annotated[
        _ContentKind | None,
        Field(None, description="The kind of content on the page."),
    ],
}


class RAGLiteModel:
    def __init__(
        self,
        task: str = "",
        db_url: str | None = None,
        embedder: str | None = None,
        llm: str | None = None,
        *,
        use_self_query: bool = False,
        use_rerank: bool = False,
        use_hybrid_search: bool = False,
        use_agentic_rag: bool = False,
    ):
        """
        Initialize your model(s) here if necessary.

        This is the constructor for your RAGLiteModel class, where you can set up any
        required initialization steps for your model(s) to function correctly.
        """
        assert task in [
            "set",
            "comparison",
            "condition",
        ], "Task must be either 'set', 'comparison', or 'condition'."
        assert not (use_agentic_rag and use_rerank), "Agentic RAG and reranking cannot be currently used together."

        self.task = task
        self.use_self_query = use_self_query
        self.use_rerank = use_rerank
        self.use_hybrid_search = use_hybrid_search
        self.use_agentic_rag = use_agentic_rag

        # set up RAGLite configuration
        self.config = RAGLiteConfig(
            db_url=db_url or os.getenv(f"RAGLITE_DB_URL_{self.task.upper()}"),  # type: ignore
            embedder=embedder or os.getenv("RAGLITE_EMBEDDER"),  # type: ignore
            llm=llm or os.getenv("RAGLITE_LLM"),  # type: ignore
            self_query=use_self_query,
            search_method=hybrid_search if use_hybrid_search else vector_search,
        )

        if use_rerank:
            self.config = replace(
                self.config,
                reranker=(
                    Reranker(
                        "rerank-v4.0-pro",
                        model_type="cohere",
                        api_key=os.getenv("COHERE_API_KEY"),
                        verbose=0,
                    )
                ),
            )

    def ingest_documents(
        self,
        max_queries: int = -1,
        categories: list[str] = ["music", "movie", "sports", "open"],
    ) -> list[dict[str, Any]]:
        """
        Ingest documents according to the raglite setup.

        Parameters
        ----------
        max_queries : int, optional
            Max number of queries to ingest. By default -1, hence ingest all queries.
        categories : list[str], optional
            List of categories to ingest among ["finance", "music", "movie", "sports", "open"]
            Defaults to ["music", "movie", "sports", "open"].
            By default all categories are ingested.
        """
        assert max_queries >= -1, "max_queries must be -1 (ingest all) or a non-negative integer."

        from raglite import Document, insert_documents

        # Read data
        file_path = Path(os.getenv(f"EVALUATION_DATASET_PATH_{self.task.upper()}", ""))
        if not file_path.exists():
            msg = f"EVALUATION_DATASET_PATH_{self.task.upper()} is not set or does not exist."
            raise FileNotFoundError(msg)
        data = read_jsonl(file_path)

        # batch size for inserting documents
        def batched(seq, size):
            for idx in range(0, len(seq), size):
                yield seq[idx : idx + size]

        ingested_n = 0
        ingested = []
        with tqdm(
            desc="Ingesting documents...",
            total=min(len(data), max_queries) if max_queries > 0 else len(data),
        ) as pbar:
            for sample in data:
                if sample["domain"] not in categories:
                    continue
                for chunk in batched(sample["search_results"], 4):
                    docs = [
                        Document.from_text(
                            content=html_to_md(doc["page_result"]),
                            url=doc["page_url"],
                            filename=doc["page_name"],
                            last_modified=extract_year_from_last_modified(
                                doc.get("page_last_modified")
                            ),
                        )
                        for doc in chunk
                    ]
                    docs = list(expand_document_metadata(docs, METADATA_FIELDS, config=self.config, strict=False)) # type: ignore
                    insert_documents(docs, config=self.config)
                    time.sleep(0.5)  # avoid rate limiting
                ingested_n += 1
                ingested.append(sample)
                pbar.update(1)
                if 0 < max_queries == ingested_n:
                    break

        return ingested

    def get_chunks_via_rerank(self, query: str, num_chunks: int, oversample_factor: int = 4) -> list[ChunkSpan]:
        """
        Retrieve relevant chunk spans for a given query using vector search and reranking.

        Parameters
        ----------
            query (str): The user query for which relevant chunks need to be retrieved.
            num_chunks (int): The number of relevant chunk spans to retrieve.
            oversample_factor (int, optional): The factor by which to oversample candidate chunks before reranking. Defaults to 4.
        """
        from raglite import rerank_chunks, retrieve_chunk_spans, retrieve_chunks, vector_search

        # Step 1: Perform vector search to retrieve initial candidate chunks
        chunk_ids_vector, _ = vector_search(query, num_results=num_chunks * oversample_factor, config=self.config)

        # Step 2: Retrieve the content and metadata of the candidate chunks
        chunk_spans = retrieve_chunks(chunk_ids_vector, config=self.config)

        # Step 3: Rerank the retrieved chunks based on their relevance to the query
        chunks_reranked = rerank_chunks(query, chunk_spans, config=self.config)

        # Step 4: Select the top 'num_chunks' relevant chunks after reranking
        top_chunks = chunks_reranked[:num_chunks]

        # Extend chunks with their neighbors for more context
        chunk_spans = retrieve_chunk_spans(top_chunks, config=self.config)

        return chunk_spans

    def get_batch_size(self) -> int:
        """
        Determine the batch size that is used by the evaluator when calling the `batch_generate_answer` function.

        Returns
        -------
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_generate_answer calls, or stay a static value.
        """
        self.batch_size = 1
        return self.batch_size

    def batch_generate_answer(self, batch: dict[str, Any]) -> tuple[list[str], list[list]]:
        """
        Generate answers for a batch of queries using associated (pre-cached) search results and query times.

        Parameters
        ----------
            batch (dict[str, Any]): A dictionary containing a batch of input queries with the following keys:
                - 'interaction_id;  (list[str]): list of interaction_ids for the associated queries
                - 'query' (list[str]): list of user queries.
                - 'search_results' (list[list[dict]]): list of search result lists, each corresponding
                                                      to a query.
                - 'query_time' (list[str]): list of timestamps (represented as a string), each corresponding to when a query was made.

        Returns
        -------
            list[str]: A list of plain text responses for each query in the batch. Each response is limited to 75 tokens.
            If the generated response exceeds 75 tokens, it will be truncated to fit within this limit.

        Notes
        -----
        - If the correct answer is uncertain, it's preferable to respond with "I don't know" to avoid
          the penalty for hallucination.
        - Response Time: Ensure that your model processes and responds to each query within 30 seconds.
          Failing to adhere to this time constraint **will** result in a timeout during evaluation.
        """
        _ = batch["interaction_id"]
        queries = batch["query"]
        _ = batch["search_results"]
        query_times = batch["query_time"]

        answers = []
        chunks = []
        for query, query_time in tqdm(
            zip(queries, query_times, strict=True),
            desc="Batch processing...",
            leave=False,
            total=len(queries),
        ):

            messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find information from a collection of documents.\n"
                    f"Today's date is {query_time}.",
                }
            ]

            # from raglite import search_and_rerank_chunk_spans
            # replace(self.config, search_method=partial(search_and_rerank_chunk_spans, num_chunks=5, config=self.config))

            chunk_spans = []
            if self.use_agentic_rag:
                # If using agentic RAG, we start with just the user query and allow the model to iteratively retrieve chunks as needed.
                messages.append({"role": "user", "content": query})
            else:
                # If not using agentic RAG, we retrieve relevant chunks upfront (optionally with reranking) and provide them to the model in one go.
                if self.use_rerank:
                    chunk_spans = self.get_chunks_via_rerank(query=query, num_chunks=5)
                else:
                    from raglite import retrieve_context

                    chunk_spans = retrieve_context(query=query, num_chunks=5, config=self.config)

                # Add retrieved context to the message history for RAG
                messages.append(add_context(user_prompt=query, context=chunk_spans, config=self.config))

            # Stream the RAG response and append it to the message history
            stream = rag(messages, config=self.config, on_retrieval=lambda x: chunk_spans.extend(x) if self.use_agentic_rag else None)
            answer = ""
            for update in stream:
                answer += update

            answers.append(answer)
            chunks.append(chunk_spans)

        return answers, chunks

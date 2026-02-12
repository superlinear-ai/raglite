# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from crag.models.utils import html_to_md, read_jsonl

load_dotenv()


class OpenAIRAGModel:
    """
    OpenAI Assistant + File Search backed model.
    """

    def __init__(
        self,
        task: str = "",
        vector_store_id: str | None = None,
        model: str | None = None,
    ):
        # set up the OpenAI client
        self.model = model or os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-5-mini")
        self.client = OpenAI(
            base_url=os.getenv("AZURE_API_BASE", "") + "openai/v1/",
            api_key=os.getenv("AZURE_API_KEY"),
        )

        # set task
        assert task in ["set", "comparison"], "Task must be either 'set' or 'comparison'."
        self.task = task

        # create an assistant for CRAG evaluation
        if vector_store_id:
            vss = self.client.vector_stores.list()  # ensure vector stores are accessible
            if vector_store_id not in [vs.id for vs in vss.data]:
                raise ValueError(f"Vector store provided {vector_store_id} does not exist.")
            self.vector_store = self.client.vector_stores.retrieve(vector_store_id)
            print(f"Using existing vector store with ID: {self.vector_store.id}")
        else:
            print("Creating new vector store...")
            # create vector store if not existing
            self.vector_store = self.client.vector_stores.create(
                name=f"crag-dataset-{self.task}",
            )
            print(f"Created vector store with ID: {self.vector_store.id}")

    def ingest_documents(
        self,
        max_queries: int = -1,
        categories: list[str] = ["music", "movie", "sports", "open"],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Ingest documents from the dataset into OpenAI Files and create a vector store."""
        # Read data
        file_path = Path(os.getenv(f"EVALUATION_DATASET_PATH_{self.task.upper()}", ""))
        if not file_path.exists():
            raise FileNotFoundError(
                f"EVALUATION_DATASET_PATH_{self.task.upper()} is not set or does not exist."
            )
        data = read_jsonl(file_path)

        # upload documents to OpenAI Files
        ingested_n = 0
        ingested = []
        file_ids: list[str] = []

        with tqdm(
            desc="Uploading documents to OpenAI Files...",
            total=min(len(data), max_queries) if max_queries > 0 else len(data),
        ) as pbar:
            for sample in data:
                if sample["domain"] not in categories:
                    continue
                for doc in sample["search_results"]:
                    content = html_to_md(doc["page_result"])
                    header = (
                        f"URL: {doc['page_url']}\nLast-Modified: {doc['page_last_modified']}\n\n"
                    )
                    payload = (header + content).encode("utf-8")

                    # upload to OpenAI Files via in-memory buffer
                    buf = io.BytesIO(payload)
                    buf.name = f"{doc['page_name'] or 'document'}.md"
                    uploaded = self.client.files.create(file=buf, purpose="assistants")
                    file_ids.append(uploaded.id)

                    # batch add files to vector store
                    if len(file_ids) >= 50:
                        self._add_files_to_vector_store(self.vector_store.id, file_ids)
                        file_ids = []
                ingested_n += 1
                ingested.append(sample)
                pbar.update(1)
                if 0 < max_queries == ingested_n:
                    break

        # add remaining files
        if file_ids:
            self._add_files_to_vector_store(self.vector_store.id, file_ids)

        return ingested, file_ids

    def _add_files_to_vector_store(self, vector_store_id: str, file_ids: list[str]) -> None:
        """Add files to the vector store and wait for processing to complete."""
        for file_id in file_ids:
            self.client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=file_id)

    def get_batch_size(self) -> int:
        self.batch_size = 1
        return self.batch_size

    def extract_file_search_results_from_response(self, response: Any) -> list[Any]:
        results: list[Any] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "file_search_call":
                continue
            for result in getattr(item, "results", []) or []:
                results.append(result)
        return results

    def batch_generate_answer(self, batch: dict[str, Any]) -> tuple[list[str], list[list]]:
        _ = batch["interaction_id"]
        queries = batch["query"]
        _ = batch["search_results"]
        query_times = batch["query_time"]

        answers: list[str] = []
        chunks: list[list] = []
        for query, query_time in tqdm(
            zip(queries, query_times),
            desc="Batch processing...",
            leave=False,
            total=len(queries),
        ):

            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": "You are an AI assistant that helps people find information "
                        "from a collection of documents. Provide concise, accurate, and "
                        "helpful answers based on the context provided by the file search results.\n\n"
                        f"Today is {query_time}",
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [self.vector_store.id],
                        "max_num_results": 5,
                    }
                ],
                tool_choice="required",
                max_tool_calls=1,
            )
            answers.append(response.output_text)
            chunks.append(self.extract_file_search_results_from_response(response))

        return answers, chunks

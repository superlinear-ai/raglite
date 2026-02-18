"""Run self-query extraction for questions from a JSONL file."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from crag.models.utils import read_jsonl
from raglite import RAGLiteConfig, hybrid_search, keyword_search, vector_search
from raglite._search import _self_query

logger = logging.getLogger(__name__)

SEARCH_METHODS = {
    "vector": vector_search,
    "keyword": keyword_search,
    "hybrid": hybrid_search,
}


def _parse_indices(raw_indices: str) -> list[int]:
    indices: list[int] = []
    for raw_token in raw_indices.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                error_message = f"Invalid range '{token}': end must be >= start."
                raise ValueError(error_message)
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))
    return sorted(set(indices))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-url",
        default=os.getenv("RAGLITE_DB_URL_COMPARISON", ""),
        help="Database URL used by self-query metadata lookup.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/crag_comparison_150.jsonl"),
        help="Path to CRAG JSONL file.",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("RAGLITE_LLM", "azure/gpt-5-mini"),
        help="LLM used by self-query extraction.",
    )
    parser.add_argument(
        "--embedder",
        default=os.getenv("RAGLITE_EMBEDDER", "azure/text-embedding-3-large"),
        help="Embedder placeholder for config initialization.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of questions to process.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start offset after filtering rows.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Optional domain filter (exact match).",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional row indices, e.g. '0,5,10-12'. Uses dataset absolute indices.",
    )
    parser.add_argument(
        "--search-check",
        action="store_true",
        help="Also run search with the extracted metadata filter and show hit count.",
    )
    parser.add_argument(
        "--search-method",
        choices=tuple(SEARCH_METHODS),
        default="vector",
        help="Search method used only when --search-check is set.",
    )
    parser.add_argument(
        "--num-results",
        type=int,
        default=5,
        help="Requested number of results for --search-check.",
    )
    return parser.parse_args()


def _select_rows(rows: list[dict[str, object]], args: argparse.Namespace) -> list[tuple[int, dict[str, object]]]:
    indexed_rows = list(enumerate(rows))
    if args.domain:
        indexed_rows = [
            (row_index, row)
            for row_index, row in indexed_rows
            if str(row.get("domain", "")) == args.domain
        ]

    if args.indices:
        selected_indices = set(_parse_indices(args.indices))
        indexed_rows = [
            (row_index, row) for row_index, row in indexed_rows if row_index in selected_indices
        ]
    else:
        indexed_rows = indexed_rows[args.offset : args.offset + args.limit]

    if args.limit >= 0:
        indexed_rows = indexed_rows[: args.limit]
    return indexed_rows


def main() -> None:
    """Run self-query tests for selected questions."""
    load_dotenv(dotenv_path=Path(".env"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    if not args.db_url:
        error_message = (
            "Missing --db-url and RAGLITE_DB_URL_COMPARISON is not set. "
            "Provide a database URL."
        )
        raise ValueError(error_message)
    if not args.dataset_path.exists():
        error_message = f"Dataset not found: {args.dataset_path}"
        raise FileNotFoundError(error_message)
    if args.limit == 0:
        logger.info("Nothing to do: --limit is 0.")
        return

    rows = read_jsonl(args.dataset_path)
    selected_rows = _select_rows(rows, args)

    logger.info("Total rows in dataset: %d", len(rows))
    logger.info("Selected rows: %d", len(selected_rows))

    config = RAGLiteConfig(
        db_url=args.db_url,
        llm=args.llm,
        embedder=args.embedder,
        reranker=None,
        self_query=True,
    )
    search_config = replace(config, self_query=False)
    search_method = SEARCH_METHODS[args.search_method]

    for row_index, row in selected_rows:
        query = str(row.get("query", "")).strip()
        if not query:
            logger.info("row=%d interaction_id=%s skipped: empty query", row_index, row.get("interaction_id"))
            continue

        metadata_filter = _self_query(query, config=config)
        logger.info(
            "row=%d interaction_id=%s domain=%s question_type=%s",
            row_index,
            row.get("interaction_id"),
            row.get("domain"),
            row.get("question_type"),
        )
        logger.info("query=%s", query)
        logger.info("self_query_filter=%s", metadata_filter)

        if args.search_check:
            chunk_ids, _ = search_method(
                query,
                num_results=args.num_results,
                metadata_filter=metadata_filter,
                config=search_config,
            )
            logger.info(
                "search_check method=%s requested=%d returned=%d",
                args.search_method,
                args.num_results,
                len(chunk_ids),
            )

        logger.info("---")


if __name__ == "__main__":
    main()

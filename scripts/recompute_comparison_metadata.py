"""Recompute document metadata in-place without re-indexing chunks or embeddings.

Usage
-----
Dry-run (recommended first):
    uv run python scripts/recompute_comparison_metadata.py \
      --db-url postgresql://raglite_user:raglite_password@localhost:5432/\
raglite_comparison_meta_v2 \
      --dataset-path data/crag_comparison_150.jsonl

Apply changes:
    uv run python scripts/recompute_comparison_metadata.py \
      --db-url postgresql://raglite_user:raglite_password@localhost:5432/\
raglite_comparison_meta_v2 \
      --dataset-path data/crag_comparison_150.jsonl \
      --llm gpt-4o-mini \
      --apply
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from pydantic import Field
from sqlalchemy import create_engine, delete, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select
from tqdm.auto import tqdm

from crag.models.utils import html_to_md, read_jsonl
from raglite import Document, RAGLiteConfig, expand_document_metadata
from raglite._database import Chunk, Metadata, _adapt_metadata

logger = logging.getLogger(__name__)


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
            "Example entities: 'lebron james', 'guitar pedals', 'hiking', 'national park'.",
        ),
    ],
    "content_type": Annotated[
        _ContentKind | None,
        Field(None, description="The kind of content on the page."),
    ],
}

METADATA_EXCLUDED_FIELDS = {"filename", "uri", "url", "size", "created", "modified"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", required=True, help="Target PostgreSQL DB URL.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/crag_comparison_150.jsonl"),
        help="Path to CRAG comparison JSONL.",
    )
    parser.add_argument(
        "--llm",
        default=os.getenv("RAGLITE_LLM", "gpt-4o-mini"),
        help="LLM used by expand_document_metadata.",
    )
    parser.add_argument(
        "--embedder",
        default=os.getenv("RAGLITE_EMBEDDER", "text-embedding-3-small"),
        help="Embedder placeholder for config initialization (not used for metadata extraction).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of documents per metadata extraction batch.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of documents to process.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, script runs in dry-run mode.",
    )
    return parser.parse_args()


def _create_database_engine(database_url: str) -> Engine:
    parsed_url = make_url(database_url)
    if parsed_url.get_backend_name() == "postgresql" and "+" not in parsed_url.drivername:
        parsed_url = parsed_url.set(drivername="postgresql+pg8000")
    return create_engine(parsed_url)


def _normalize_metadata_value(value: Any) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        normalized_value = value.strip()
        return normalized_value if normalized_value else None
    if isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, list[str | int | float | bool]]:
    normalized_metadata: dict[str, list[str | int | float | bool]] = {}
    for metadata_name, metadata_value in metadata.items():
        values = metadata_value if isinstance(metadata_value, list) else [metadata_value]
        normalized_values: list[str | int | float | bool] = []
        for value in values:
            normalized_value = _normalize_metadata_value(value)
            if normalized_value is None:
                continue
            if normalized_value not in normalized_values:
                normalized_values.append(normalized_value)
        if normalized_values:
            normalized_metadata[metadata_name] = normalized_values
    return normalized_metadata


def _url_key_for_match(url: str | None) -> str:
    return "" if url is None else str(url)


def _filename_key_for_match(filename: str | None) -> str:
    return "" if filename is None else str(filename)


def _build_documents_from_dataset(dataset_path: Path) -> tuple[dict[str, Document], int]:
    dataset = read_jsonl(dataset_path)
    documents_by_id: dict[str, Document] = {}
    skipped_empty_count = 0
    for sample in tqdm(dataset, desc="Building document map from JSONL", leave=False):
        for page in sample.get("search_results", []):
            markdown_content = html_to_md(page["page_result"])
            if not markdown_content.strip():
                skipped_empty_count += 1
                continue
            document = Document.from_text(
                content=markdown_content,
                url=page.get("page_url"),
                filename=page.get("page_name"),
            )
            if document.id not in documents_by_id:
                documents_by_id[document.id] = document
    return documents_by_id, skipped_empty_count


def _extract_recomputed_metadata(
    document_matches: list[tuple[str, Document]],
    *,
    config: RAGLiteConfig,
    batch_size: int,
) -> dict[str, dict[str, list[str | int | float | bool]]]:
    metadata_by_document_id: dict[str, dict[str, list[str | int | float | bool]]] = {}
    for index in tqdm(
        range(0, len(document_matches), batch_size),
        desc="Extracting metadata",
        leave=False,
    ):
        batch_matches = document_matches[index : index + batch_size]
        db_document_ids = [db_document_id for db_document_id, _ in batch_matches]
        document_batch = [source_document for _, source_document in batch_matches]
        expanded_batch = list(
            expand_document_metadata(document_batch, METADATA_FIELDS, config=config, strict=False)
        )
        for db_document_id, expanded_document in zip(db_document_ids, expanded_batch, strict=True):
            metadata_by_document_id[db_document_id] = _normalize_metadata(
                expanded_document.metadata_
            )
    return metadata_by_document_id


def _reset_database_metadata(session: Session) -> None:
    session.execute(delete(Metadata))
    for document in session.exec(select(Document)).all():
        document.metadata_ = {}
        flag_modified(document, "metadata_")
    for chunk in session.exec(select(Chunk)).all():
        chunk.metadata_ = {}
        flag_modified(chunk, "metadata_")
    session.flush()


def _update_document_and_chunk_metadata(
    session: Session,
    document_id: str,
    metadata: dict[str, list[str | int | float | bool]],
) -> dict[str, set[str | int | float | bool]]:
    metadata_values_by_name: dict[str, set[str | int | float | bool]] = defaultdict(set)
    document = session.get(Document, document_id)
    if document is None:
        return metadata_values_by_name

    document.metadata_ = metadata
    flag_modified(document, "metadata_")

    chunk_metadata_base: dict[str, Any] = {"filename": document.filename}
    if document.url:
        chunk_metadata_base["url"] = document.url
    chunk_metadata_base.update(metadata)
    chunk_metadata = _adapt_metadata(chunk_metadata_base)
    for chunk in session.exec(select(Chunk).where(Chunk.document_id == document.id)).all():
        chunk.metadata_ = chunk_metadata
        flag_modified(chunk, "metadata_")

    for metadata_name, values in metadata.items():
        if metadata_name in METADATA_EXCLUDED_FIELDS:
            continue
        for value in values:
            metadata_values_by_name[metadata_name].add(value)
    return metadata_values_by_name


def _apply_recomputed_metadata(
    session: Session,
    metadata_by_document_id: dict[str, dict[str, list[str | int | float | bool]]],
) -> None:
    _reset_database_metadata(session)

    metadata_values_by_name: dict[str, set[str | int | float | bool]] = defaultdict(set)
    for document_id, metadata in tqdm(
        metadata_by_document_id.items(),
        desc="Updating document/chunk metadata",
        leave=False,
    ):
        updated_values = _update_document_and_chunk_metadata(session, document_id, metadata)
        for metadata_name, values in updated_values.items():
            metadata_values_by_name[metadata_name].update(values)

    for metadata_name, values in metadata_values_by_name.items():
        session.add(Metadata(name=metadata_name, values=sorted(values, key=str)))
    session.commit()


def _log_database_summary(engine: Engine) -> None:
    with engine.connect() as connection:
        document_count = connection.execute(text("SELECT COUNT(*) FROM document")).scalar_one()
        chunk_count = connection.execute(text("SELECT COUNT(*) FROM chunk")).scalar_one()
        metadata_rows_count = connection.execute(text("SELECT COUNT(*) FROM metadata")).scalar_one()
        empty_document_metadata_count = connection.execute(
            text("SELECT COUNT(*) FROM document WHERE metadata::jsonb = '{}'::jsonb")
        ).scalar_one()
    logger.info(
        "Database summary: documents=%d, chunks=%d, metadata_rows=%d, empty_document_metadata=%d",
        document_count,
        chunk_count,
        metadata_rows_count,
        empty_document_metadata_count,
    )


def _build_dataset_lookups(
    dataset_documents: list[Document],
) -> tuple[
    dict[tuple[str, str], list[Document]],
    dict[str, list[Document]],
    dict[str, list[Document]],
]:
    dataset_by_url_filename: dict[tuple[str, str], list[Document]] = defaultdict(list)
    dataset_by_url: dict[str, list[Document]] = defaultdict(list)
    dataset_by_filename: dict[str, list[Document]] = defaultdict(list)
    for document in dataset_documents:
        normalized_url = _url_key_for_match(document.url)
        normalized_filename = _filename_key_for_match(document.filename)
        if normalized_url:
            dataset_by_url[normalized_url].append(document)
        if normalized_filename:
            dataset_by_filename[normalized_filename].append(document)
        if normalized_url and normalized_filename:
            dataset_by_url_filename[(normalized_url, normalized_filename)].append(document)
    return dataset_by_url_filename, dataset_by_url, dataset_by_filename


def _get_unique_candidate(candidates: list[Document]) -> Document | None:
    return candidates[0] if len(candidates) == 1 else None


def _match_document_by_fallbacks(
    db_filename: str,
    db_url: str | None,
    *,
    dataset_by_url_filename: dict[tuple[str, str], list[Document]],
    dataset_by_url: dict[str, list[Document]],
    dataset_by_filename: dict[str, list[Document]],
) -> tuple[str | None, Document | None]:
    normalized_url = _url_key_for_match(db_url)
    normalized_filename = _filename_key_for_match(db_filename)

    if normalized_url and normalized_filename:
        matched_document = _get_unique_candidate(
            dataset_by_url_filename.get((normalized_url, normalized_filename), [])
        )
        if matched_document is not None:
            return "url_filename", matched_document

    if normalized_url:
        matched_document = _get_unique_candidate(dataset_by_url.get(normalized_url, []))
        if matched_document is not None:
            return "url", matched_document

    if normalized_filename:
        matched_document = _get_unique_candidate(dataset_by_filename.get(normalized_filename, []))
        if matched_document is not None:
            return "filename", matched_document

    return None, None


def _build_document_matches(
    session: Session,
    dataset_documents_by_id: dict[str, Document],
) -> tuple[
    list[tuple[str, Document]],
    set[str],
    dict[str, int],
]:
    dataset_documents = list(dataset_documents_by_id.values())
    dataset_by_url_filename, dataset_by_url, dataset_by_filename = _build_dataset_lookups(
        dataset_documents
    )

    matching_counts = {
        "id": 0,
        "url_filename": 0,
        "url": 0,
        "filename": 0,
        "unmatched": 0,
    }
    unmatched_document_ids: set[str] = set()
    document_matches: list[tuple[str, Document]] = []

    database_documents = session.exec(select(Document.id, Document.filename, Document.url)).all()
    for db_document_id, db_filename, db_url in database_documents:
        if db_document_id in dataset_documents_by_id:
            document_matches.append((db_document_id, dataset_documents_by_id[db_document_id]))
            matching_counts["id"] += 1
            continue

        match_kind, matched_document = _match_document_by_fallbacks(
            db_filename,
            db_url,
            dataset_by_url_filename=dataset_by_url_filename,
            dataset_by_url=dataset_by_url,
            dataset_by_filename=dataset_by_filename,
        )
        if matched_document is not None and match_kind is not None:
            document_matches.append((db_document_id, matched_document))
            matching_counts[match_kind] += 1
        else:
            unmatched_document_ids.add(db_document_id)
            matching_counts["unmatched"] += 1

    return document_matches, unmatched_document_ids, matching_counts


def main() -> None:
    """Recompute metadata for existing documents using LLM extraction."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if not args.dataset_path.exists():
        error_message = f"Dataset not found: {args.dataset_path}"
        raise FileNotFoundError(error_message)
    if args.batch_size <= 0:
        error_message = "--batch-size must be greater than 0."
        raise ValueError(error_message)

    engine = _create_database_engine(args.db_url)
    with Session(engine) as session:
        database_document_ids = set(session.exec(select(Document.id)).all())

    documents_by_id_from_dataset, skipped_empty_count = _build_documents_from_dataset(
        args.dataset_path
    )
    dataset_document_ids = set(documents_by_id_from_dataset)
    with Session(engine) as session:
        document_matches, unmatched_document_ids, matching_counts = _build_document_matches(
            session, documents_by_id_from_dataset
        )

    if args.limit is not None:
        document_matches = document_matches[: args.limit]

    logger.info("Dataset unique documents: %d", len(dataset_document_ids))
    logger.info("Skipped empty markdown rows from JSONL: %d", skipped_empty_count)
    logger.info("Database documents: %d", len(database_document_ids))
    logger.info("Matching documents (with fallbacks): %d", len(document_matches))
    logger.info(
        "Matched by id=%d, url+filename=%d, url=%d, filename=%d",
        matching_counts["id"],
        matching_counts["url_filename"],
        matching_counts["url"],
        matching_counts["filename"],
    )
    logger.info(
        "Not matched=%d",
        len(unmatched_document_ids),
    )
    logger.info("Only in dataset: %d", len(dataset_document_ids - database_document_ids))
    logger.info(
        "Only in database by id (before fallback): %d",
        len(database_document_ids - dataset_document_ids),
    )
    _log_database_summary(engine)

    if not args.apply:
        logger.info("Dry-run complete. Re-run with --apply to recompute and write metadata.")
        return

    config = RAGLiteConfig(
        db_url="duckdb:///:memory:",
        llm=args.llm,
        embedder=args.embedder,
        reranker=None,
    )
    metadata_by_document_id = _extract_recomputed_metadata(
        document_matches,
        config=config,
        batch_size=args.batch_size,
    )

    with Session(engine) as session:
        _apply_recomputed_metadata(session, metadata_by_document_id)

    _log_database_summary(engine)


if __name__ == "__main__":
    main()

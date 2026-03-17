r"""Ingest CRAG HTML pages into a RAGLite PostgreSQL database for the demo.

Usage
-----
uv run python scripts/ingest_crag.py \\
    --dataset-path data/crag_sample.jsonl \\
    --db-url postgresql://user:pass@localhost/raglite_demo \\
    --llm gpt-4o-mini \\
    --categories music movie sports open

The script:
1. Reads a CRAG JSONL file (one JSON object per line).
2. Converts each search-result page from HTML to Markdown via markdownify.
3. Inserts deduplicated documents into the database (idempotent: same URL = same id).
4. Runs expand_document_metadata to populate domains / primary_entity / content_type.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal

import bs4
import dotenv
import markdownify as md
from pydantic import Field
from tqdm import tqdm

from raglite import RAGLiteConfig

YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata schema (mirrors bench/mm-crag branch)
# ---------------------------------------------------------------------------

_Domain = Literal[
    "sports",
    "music",
    "movie",
    "popculture",
    "forums",
    "travel_geography",
    "technology",
    "health_nutrition",
    "science_research",
    "other",
]

_ContentKind = Literal[
    "profile_bio",
    "reference_explainer",
    "news_article",
    "stats_scores",
    "comparison",
    "ranking_list",
    "review",
    "guide_howto",
    "forum_thread",
    "other",
]

METADATA_FIELDS: dict[str, Any] = {
    "domains": Annotated[
        list[_Domain] | None,
        Field(
            default_factory=list,
            max_length=3,
            description="Broad topic areas covered by this page (max 3).",
        ),
    ],
    "primary_entity": Annotated[
        list[str] | None,
        Field(
            default_factory=list,
            max_length=2,
            description=(
                "Main subjects of the page in lowercase (max 2). "
                "Examples: 'lebron james', 'guitar pedals', 'national park'."
            ),
        ),
    ],
    "content_type": Annotated[
        _ContentKind | None,
        Field(None, description="The kind of content on the page."),
    ],
}


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------


def extract_year_from_last_modified(last_modified: str | None) -> str:
    """Extract a 4-digit year from an HTTP-style Last-Modified string."""
    if last_modified is None:
        return ""
    normalized_last_modified = str(last_modified).strip()
    if not normalized_last_modified:
        return ""
    year_match = YEAR_PATTERN.search(normalized_last_modified)
    if year_match is None:
        return ""
    return year_match.group(0)


def denoise_html(html: str | bytes) -> str:
    """Return a cleaned HTML string (keeps markup) instead of plain text."""
    html_input = html.decode("utf-8", errors="replace") if isinstance(html, bytes) else html or ""

    soup = bs4.BeautifulSoup(html_input, "lxml")

    # Remove the obvious high-noise elements
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    # Remove common boilerplate containers
    for t in soup.find_all(["nav", "footer", "aside", "header"]):
        t.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda s: isinstance(s, bs4.Comment)):
        comment.extract()

    # Serialize and collapse excessive blank lines to avoid huge whitespace runs
    cleaned_html = str(soup)
    cleaned_html = re.sub(r"\n{3,}", "\n\n", cleaned_html)

    return cleaned_html


def html_to_md(html: str | bytes, *, save: bool = False) -> str:
    """Convert HTML to Markdown."""
    cleaned_html = denoise_html(html)
    markdown_text = md.markdownify(cleaned_html, strip=["a"])
    if save:
        with open("custom.md", "w", encoding="utf-8") as f:  # noqa: PTH123
            f.write(markdown_text)
    return markdown_text


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL file"):
            data.append(json.loads(line))  # noqa: PERF401
    return data


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def ingest(dataset_path: Path, config: RAGLiteConfig) -> None:
    """Ingest CRAG JSONL into a RAGLite database with metadata expansion."""
    from raglite import Document, insert_documents
    from raglite._extract import expand_document_metadata

    # read data
    data = read_jsonl(dataset_path)
    logger.info("Loaded %d records from %s", len(data), dataset_path)

    # batch size for inserting documents
    def batched(seq, size):
        for idx in range(0, len(seq), size):
            yield seq[idx : idx + size]

    # insert documents
    with tqdm(desc="Ingesting documents...", total=len(data)) as pbar:
        for sample in data:
            for chunk in batched(sample["search_results"], 5):
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
                docs = list(
                    expand_document_metadata(docs, METADATA_FIELDS, config=config, strict=False)
                )
                insert_documents(docs, config=config)
                time.sleep(0.5)  # avoid rate limiting
            pbar.update(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest CRAG HTML pages into a RAGLite PostgreSQL database."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the CRAG JSONL file.",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="SQLAlchemy database URL (e.g. postgresql://user:pass@host/db).",
    )
    parser.add_argument(
        "--embedder",
        required=True,
        help="Embedder to use for metadata expansion (default: azure/text-embedding-3-large).",
    )
    parser.add_argument(
        "--llm",
        default="azure/gpt-5-mini",
        help="LLM to use for metadata expansion (default: azure/gpt-5-mini).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if not args.dataset_path.exists():
        logger.error("Dataset file not found: %s", args.dataset_path)
        sys.exit(1)

    config = RAGLiteConfig(db_url=args.db_url, llm=args.llm, embedder=args.embedder)
    logger.info("Target database: %s", args.db_url)
    logger.info("LLM for metadata expansion: %s", args.llm)
    ingest(dataset_path=args.dataset_path, config=config)

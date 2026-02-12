# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from pathlib import Path
from typing import Any

import bs4
import markdownify as md
from tqdm import tqdm


def trim_predictions_to_max_token_length(prediction: str) -> str:
    """Trim prediction output to approximately 75 tokens using whitespace splitting."""
    max_token_length = 75
    token_avg_ch = 4  # Average characters per token
    prediction_words = str(prediction).split()
    return " ".join(prediction_words[: max_token_length * token_avg_ch])


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
        with open("custom.md", "w", encoding="utf-8") as f:
            f.write(markdown_text)
    return markdown_text


def read_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries."""
    data: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL file"):
            data.extend(json.loads(line))
    return data

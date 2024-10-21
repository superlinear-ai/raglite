"""Convert any document to Markdown."""

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import mdformat
import numpy as np
from pdftext.extraction import dictionary_output
from sklearn.cluster import KMeans


def parsed_pdf_to_markdown(pages: list[dict[str, Any]]) -> list[str]:  # noqa: C901, PLR0915
    """Convert a PDF parsed with pdftext to Markdown."""

    def add_heading_level_metadata(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:  # noqa: C901
        """Add heading level metadata to a PDF parsed with pdftext."""

        def extract_font_size(span: dict[str, Any]) -> float:
            """Extract the font size from a text span."""
            font_size: float = 1.0
            if span["font"]["size"] > 1:  # A value of 1 appears to mean "unknown" in pdftext.
                font_size = span["font"]["size"]
            elif digit_sequences := re.findall(r"\d+", span["font"]["name"] or ""):
                font_size = float(digit_sequences[-1])
            elif "\n" not in span["text"]:  # Occasionally a span can contain a newline character.
                if round(span["rotation"]) in (0.0, 180.0, -180.0):
                    font_size = span["bbox"][3] - span["bbox"][1]
                elif round(span["rotation"]) in (90.0, -90.0, 270.0, -270.0):
                    font_size = span["bbox"][2] - span["bbox"][0]
            return font_size

        # Copy the pages.
        pages = deepcopy(pages)
        # Extract an array of all font sizes used by the text spans.
        font_sizes = np.asarray(
            [
                extract_font_size(span)
                for page in pages
                for block in page["blocks"]
                for line in block["lines"]
                for span in line["spans"]
            ]
        )
        font_sizes = np.round(font_sizes * 2) / 2
        unique_font_sizes, counts = np.unique(font_sizes, return_counts=True)
        # Determine the paragraph font size as the mode font size.
        tiny = unique_font_sizes < min(5, np.max(unique_font_sizes))
        counts[tiny] = -counts[tiny]
        mode = np.argmax(counts)
        counts[tiny] = -counts[tiny]
        mode_font_size = unique_font_sizes[mode]
        # Determine (at most) 6 heading font sizes by clustering font sizes larger than the mode.
        heading_font_sizes = unique_font_sizes[mode + 1 :]
        if len(heading_font_sizes) > 0:
            heading_counts = counts[mode + 1 :]
            kmeans = KMeans(n_clusters=min(6, len(heading_font_sizes)), random_state=42)
            kmeans.fit(heading_font_sizes[:, np.newaxis], sample_weight=heading_counts)
            heading_font_sizes = np.sort(np.ravel(kmeans.cluster_centers_))[::-1]
        # Add heading level information to the text spans and lines.
        for page in pages:
            for block in page["blocks"]:
                for line in block["lines"]:
                    if "md" not in line:
                        line["md"] = {}
                    heading_level = np.zeros(8)  # 0-5: <h1>-<h6>, 6: <p>, 7: <small>
                    for span in line["spans"]:
                        if "md" not in span:
                            span["md"] = {}
                        span_font_size = extract_font_size(span)
                        if span_font_size < mode_font_size:
                            idx = 7
                        elif span_font_size == mode_font_size:
                            idx = 6
                        else:
                            idx = np.argmin(np.abs(heading_font_sizes - span_font_size))  # type: ignore[assignment]
                        span["md"]["heading_level"] = idx + 1
                        heading_level[idx] += len(span["text"])
                    line["md"]["heading_level"] = np.argmax(heading_level) + 1
        return pages

    def add_emphasis_metadata(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add emphasis metadata such as bold and italic to a PDF parsed with pdftext."""
        # Copy the pages.
        pages = deepcopy(pages)
        # Add emphasis metadata to the text spans.
        for page in pages:
            for block in page["blocks"]:
                for line in block["lines"]:
                    if "md" not in line:
                        line["md"] = {}
                    for span in line["spans"]:
                        if "md" not in span:
                            span["md"] = {}
                        span["md"]["bold"] = span["font"]["weight"] > 500  # noqa: PLR2004
                        span["md"]["italic"] = "ital" in (span["font"]["name"] or "").lower()
                    line["md"]["bold"] = all(
                        span["md"]["bold"] for span in line["spans"] if span["text"].strip()
                    )
                    line["md"]["italic"] = all(
                        span["md"]["italic"] for span in line["spans"] if span["text"].strip()
                    )
        return pages

    def strip_page_numbers(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip page numbers from a PDF parsed with pdftext."""
        # Copy the pages.
        pages = deepcopy(pages)
        # Remove lines that only contain a page number.
        for page in pages:
            for block in page["blocks"]:
                block["lines"] = [
                    line
                    for line in block["lines"]
                    if not re.match(
                        r"^\s*[#0]*\d+\s*$", "".join(span["text"] for span in line["spans"])
                    )
                ]
        return pages

    def convert_to_markdown(pages: list[dict[str, Any]]) -> list[str]:  # noqa: C901, PLR0912
        """Convert a list of pages to Markdown."""
        pages_md = []
        for page in pages:
            page_md = ""
            for block in page["blocks"]:
                block_text = ""
                for line in block["lines"]:
                    # Build the line text and style the spans.
                    line_text = ""
                    for span in line["spans"]:
                        if (
                            not line["md"]["bold"]
                            and not line["md"]["italic"]
                            and span["md"]["bold"]
                            and span["md"]["italic"]
                        ):
                            line_text += f"***{span['text']}***"
                        elif not line["md"]["bold"] and span["md"]["bold"]:
                            line_text += f"**{span['text']}**"
                        elif not line["md"]["italic"] and span["md"]["italic"]:
                            line_text += f"*{span['text']}*"
                        else:
                            line_text += span["text"]
                    # Add emphasis to the line (if it's not a heading or whitespace).
                    line_text = line_text.rstrip()
                    line_is_whitespace = not line_text.strip()
                    line_is_heading = line["md"]["heading_level"] <= 6  # noqa: PLR2004
                    if not line_is_heading and not line_is_whitespace:
                        if line["md"]["bold"] and line["md"]["italic"]:
                            line_text = f"***{line_text}***"
                        elif line["md"]["bold"]:
                            line_text = f"**{line_text}**"
                        elif line["md"]["italic"]:
                            line_text = f"*{line_text}*"
                    # Set the heading level.
                    if line_is_heading and not line_is_whitespace:
                        line_text = f"{'#' * line['md']['heading_level']} {line_text}"
                    line_text += "\n"
                    block_text += line_text
                block_text = block_text.rstrip() + "\n\n"
                page_md += block_text
            pages_md.append(page_md.strip())
        return pages_md

    def merge_split_headings(pages: list[str]) -> list[str]:
        """Merge headings that are split across lines."""

        def _merge_split_headings(match: re.Match[str]) -> str:
            atx_headings = [line.strip("# ").strip() for line in match.group().splitlines()]
            return f"{match.group(1)} {' '.join(atx_headings)}\n\n"

        pages_md = [
            re.sub(
                r"^(#+)[ \t]+[^\n]+\n+(?:^\1[ \t]+[^\n]+\n+)+",
                _merge_split_headings,
                page,
                flags=re.MULTILINE,
            )
            for page in pages
        ]
        return pages_md

    # Add heading level metadata.
    pages = add_heading_level_metadata(pages)
    # Add emphasis metadata.
    pages = add_emphasis_metadata(pages)
    # Strip page numbers.
    pages = strip_page_numbers(pages)
    # Convert the pages to Markdown.
    pages_md = convert_to_markdown(pages)
    # Merge headings that are split across lines.
    pages_md = merge_split_headings(pages_md)
    return pages_md


def document_to_markdown(doc_path: Path) -> str:
    """Convert any document to GitHub Flavored Markdown."""
    # Convert the file's content to GitHub Flavored Markdown.
    if doc_path.suffix == ".pdf":
        # Parse the PDF with pdftext and convert it to Markdown.
        pages = dictionary_output(doc_path, sort=True, keep_chars=False)
        doc = "\n\n".join(parsed_pdf_to_markdown(pages))
    else:
        try:
            # Use pandoc for everything else.
            import pypandoc

            doc = pypandoc.convert_file(doc_path, to="gfm")
        except ImportError as error:
            error_message = (
                "To convert files to Markdown with pandoc, please install the `pandoc` extra."
            )
            raise ImportError(error_message) from error
        except RuntimeError:
            # File format not supported, fall back to reading the text.
            doc = doc_path.read_text()
    # Improve Markdown quality.
    doc = mdformat.text(doc)
    return doc

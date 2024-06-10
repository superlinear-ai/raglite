"""Convert any document to Markdown."""

import re
from pathlib import Path

import mdformat
import numpy as np
import pypandoc
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTComponent, LTExpandableContainer, LTPage


def pdf_to_markdown(pdf_path: Path) -> str:  # noqa: C901, PLR0915
    """Convert a PDF to Markdown."""

    def get_font_sizes(element: LTComponent) -> list[float]:
        """Get a flattened list of font sizes used by the given element."""
        if hasattr(element, "__iter__"):
            return [fs for child in element for fs in get_font_sizes(child)]
        if hasattr(element, "size"):
            return [element.size]
        return []

    def analyse_layout(page: LTPage) -> LTPage:
        """Analyse the layout of a page and sort elements according to reading order.

        The algorithm first groups text boxes in to contiguous columns by searching for the next
        text box whose top left corner is near the last text box's bottom left corner. Then, the
        algorithm sorts the columns in reading order from top to bottom, and from left to right.
        """
        # Group text boxes into columns.
        lft, btm, rgt, top = 0, 1, 2, 3
        text_boxes = [text_box for text_box in page if hasattr(text_box, "get_text")]
        columns = []
        while text_boxes:
            # Sort text boxes from top to bottom.
            text_boxes = sorted(text_boxes, key=lambda e: -e.bbox[top] + 0.1 * e.bbox[rgt])
            # Start a new column with the first line.
            column = LTExpandableContainer()
            column.add(text_boxes.pop(0))
            # Add text boxes one by one to the column.
            while text_boxes:
                # Sort text boxes by the minimum of:
                # - the distance to column's last object's bottom left corner.
                # - the distance to the column's last object's top right corner.
                text_boxes = sorted(
                    text_boxes,
                    key=lambda e: min(
                        abs(e.bbox[lft] - column._objs[-1].bbox[lft])  # noqa: SLF001
                        + abs(e.bbox[top] - column._objs[-1].bbox[btm]),  # noqa: SLF001
                        abs(e.bbox[lft] - column._objs[-1].bbox[rgt])  # noqa: SLF001
                        + abs(e.bbox[top] - column._objs[-1].bbox[top]),  # noqa: SLF001
                    ),
                )
                # Check if the nearest neighbour should be part of this column.
                font_sizes = get_font_sizes(column)
                median_font_size = np.median(font_sizes) if font_sizes else 12.0
                is_vertical_neighbor = (
                    abs(text_boxes[0].bbox[lft] - column._objs[-1].bbox[lft])  # noqa: SLF001
                    <= 4 * median_font_size
                ) and (
                    abs(text_boxes[0].bbox[top] - column._objs[-1].bbox[btm])  # noqa: SLF001
                    <= 8 * median_font_size
                )
                is_horizontal_neighbor = (
                    abs(text_boxes[0].bbox[lft] - column._objs[-1].bbox[rgt])  # noqa: SLF001
                    <= 4 * median_font_size
                ) and (
                    abs(text_boxes[0].bbox[top] - column._objs[-1].bbox[top])  # noqa: SLF001
                    <= 8 * median_font_size
                )
                if is_vertical_neighbor or is_horizontal_neighbor:
                    # Add the line to the column.
                    column.add(text_boxes.pop(0))
                else:
                    # Column completed.
                    break
            # Append the column to the list of columns.
            columns.append(column)
        # Sort columns in top to bottom and left to right reading order.
        columns = sorted(columns, key=lambda e: (-round(e.bbox[top] * 4) / 4, e.bbox[lft]))
        # Update the page's layout and return it.
        page = LTPage(pageid=page.pageid, bbox=page.bbox, rotate=page.rotate)
        page.extend(columns)
        return page

    def get_header_levels(pages: list[LTPage]) -> tuple[dict[float, int], float]:
        """Compute a map from font size to Markdown header level."""
        font_sizes = get_font_sizes(pages)
        unique_font_sizes, counts = np.unique(font_sizes, return_counts=True)
        mode_font_size = unique_font_sizes[np.argmax(counts)] if font_sizes else 12.0
        mode_font_size = np.round(mode_font_size * 2) / 2
        top_font_sizes = {
            fs: i + 1
            for i, fs in enumerate(np.unique(np.round(np.asarray(font_sizes) * 2) / 2)[::-1][:6])
            if fs > mode_font_size
        }
        header_levels = {
            fs: top_font_sizes[np.round(fs * 2) / 2]
            for fs in set(font_sizes)
            if np.round(fs * 2) / 2 in top_font_sizes
        }
        return header_levels, mode_font_size

    def convert_to_markdown(
        element: LTComponent, header_levels: dict[float, int], mode_font_size: float = 12.0
    ) -> list[str]:
        if hasattr(element, "get_text"):
            # Get the element's text.
            text = element.get_text()
            # If the text is empty, return an empty list.
            if not text:
                return []
            # Convert the text to a Markdown header if applicable.
            max_font_size = np.max(get_font_sizes(element))  # TODO: Mode?
            if max_font_size in header_levels and not re.match(r"\s*\d+\s*", text):
                header_level = header_levels[max_font_size]
                return [
                    "\n"
                    + ("#" * header_level)
                    + " "
                    + re.sub(r"\s*\n+\s*", " ", text.strip())
                    + "\n\n"
                ]
            # Convert the text to a details element if applicable.
            if round(max_font_size * 2) / 2 < mode_font_size:
                prefix, body, suffix = re.match(
                    r"^(\s*)(\S.*?\S|\S?)(\s*)$", text, re.DOTALL
                ).groups()
                return [prefix + "<details>\n" + body + "\n</details>" + suffix]
            # Otherwise, return the text as is.
            return [text]
        if hasattr(element, "__iter__"):
            return [
                md
                for child in element
                for md in convert_to_markdown(child, header_levels, mode_font_size)
            ]
        return []

    # Parse the PDF into layout elements with pdfminer:
    # - We disable pdfminer's layout analysis because we'll do our own layout analysis.
    # - We do enable vertical text detection because it's off by default.
    laparams = LAParams(boxes_flow=None, detect_vertical=True, all_texts=True)
    pages = extract_pages(pdf_path, laparams=laparams)
    # Analyse the layout of each page and sort elements according to reading order.
    pages = [analyse_layout(page) for page in pages]
    # Convert to Markdown.
    header_levels, mode_font_size = get_header_levels(pages)
    doc = "".join(convert_to_markdown(pages, header_levels, mode_font_size))
    # Merge adjacent details.
    doc = re.sub(r"\s*</details>\s*<details>\s*", " ", doc)
    return doc


def format_markdown(doc: str) -> str:
    """Format Markdown document."""

    def _merge_nested_headers(match: re.Match) -> str:
        atx_header_prefix = match.group(1)
        atx_header_text = re.sub(r"\s*#+\s*", " ", match.group(2))
        return f"{atx_header_prefix}{atx_header_text}"

    def _merge_split_headers(match: re.Match) -> str:
        atx_headers = [line.strip("# ").strip() for line in match.group().splitlines()]
        return f"{match.group(1)} {' '.join(atx_headers)}\n\n"

    # Format.
    doc = mdformat.text(doc)
    # Remove page numbers.
    doc = re.sub(r"^[ \t]*#*[ ]*\d+[ \t]*\n", "", doc, flags=re.MULTILINE)
    # Remove horizontal rules.
    doc = re.sub(r"\n*(?:-{3,}|_{3,}|\*{3,})\n*", "\n\n", doc)
    # Merge nested headers (e.g. `#### I.1 #### My header`)
    doc = re.sub(r"^(#+)([^\n]+)$", _merge_nested_headers, doc, flags=re.MULTILINE)
    # Merge headers that are split across lines.
    doc = re.sub(
        r"^(#+)[ \t]+[^\n]+\n+(?:^\1[ \t]+[^\n]+\n+)+",
        _merge_split_headers,
        doc,
        flags=re.MULTILINE,
    )
    # Format.
    doc = mdformat.text(doc)
    return doc


def document_to_markdown(doc_path: Path) -> str:
    """Convert any document to GitHub Flavored Markdown."""
    # Convert the file's content to GitHub Flavored Markdown.
    if doc_path.suffix == ".pdf":
        # Use pdfminer with our own layout analysis for PDFs.
        doc = pdf_to_markdown(doc_path)
    else:
        # Use pandoc for everything else.
        doc = pypandoc.convert_file(doc_path, to="gfm")
    # Improve Markdown quality.
    doc = format_markdown(doc)
    return doc

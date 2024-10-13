"""Sentence splitter."""

import re

import spacy
from markdown_it import MarkdownIt
from spacy.language import Language


@Language.component("_mark_additional_sentence_boundaries")
def _mark_additional_sentence_boundaries(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    """Mark additional sentence boundaries in Markdown documents."""

    def get_markdown_heading_indexes(doc: str) -> list[tuple[int, int]]:
        """Get the indexes of the headings in a Markdown document."""
        md = MarkdownIt()
        tokens = md.parse(doc)
        headings = []
        lines = doc.splitlines(keepends=True)
        char_idx = [0]
        for line in lines:
            char_idx.append(char_idx[-1] + len(line))
        for token in tokens:
            if token.type == "heading_open":
                start_line, end_line = token.map  # type: ignore[misc]
                heading_start = char_idx[start_line]
                heading_end = char_idx[end_line]
                headings.append((heading_start, heading_end))
        return headings

    headings = get_markdown_heading_indexes(doc.text)
    for heading_start, heading_end in headings:
        # Mark the start of a heading as a new sentence.
        for token in doc:
            if heading_start <= token.idx:
                token.is_sent_start = True
                break
        # Mark the end of a heading as a new sentence.
        for token in doc:
            if heading_end <= token.idx:
                token.is_sent_start = True
                break
    return doc


def split_sentences(doc: str, max_len: int | None = None) -> list[str]:
    """Split a document into sentences."""
    # Split sentences with spaCy.
    try:
        nlp = spacy.load("xx_sent_ud_sm")
    except OSError as error:
        error_message = "Please install `xx_sent_ud_sm` with `pip install https://github.com/explosion/spacy-models/releases/download/xx_sent_ud_sm-3.7.0/xx_sent_ud_sm-3.7.0-py3-none-any.whl`."
        raise ImportError(error_message) from error
    nlp.add_pipe("_mark_additional_sentence_boundaries", before="senter")
    sentences = [sent.text_with_ws for sent in nlp(doc).sents if sent.text.strip()]
    # Apply additional splits on paragraphs and sentences because spaCy's splitting is not perfect.
    if max_len is not None:
        for pattern in (r"(?<=\n\n)", r"(?<=\.\s)"):
            sentences = [
                part
                for sent in sentences
                for part in ([sent] if len(sent) <= max_len else re.split(pattern, sent))
            ]
    # Recursively split long sentences in the middle if they are still too long.
    if max_len is not None:
        while any(len(sentence) > max_len for sentence in sentences):
            sentences = [
                part
                for sent in sentences
                for part in (
                    [sent]
                    if len(sent) <= max_len
                    else [sent[: len(sent) // 2], sent[len(sent) // 2 :]]
                )
            ]
    return sentences

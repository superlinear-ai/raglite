"""Sentence splitter."""

import spacy
from markdown_it import MarkdownIt
from spacy.language import Language


@Language.component("mark_additional_sentence_boundaries")
def mark_additional_sentence_boundaries(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
    """Mark additional sentence boundaries in Markdown documents."""

    def get_markdown_header_indexes(doc: str) -> list[tuple[int, int]]:
        """Get the indexes of the headers in a Markdown document."""
        md = MarkdownIt()
        tokens = md.parse(doc)
        headers = []
        lines = doc.splitlines(keepends=True)
        char_idx = [0]
        for line in lines:
            char_idx.append(char_idx[-1] + len(line))
        for token in tokens:
            if token.type == "heading_open":
                start_line, end_line = token.map
                header_start = char_idx[start_line]
                header_end = char_idx[end_line]
                headers.append((header_start, header_end))
        return headers

    headers = get_markdown_header_indexes(doc.text)
    for header_start, header_end in headers:
        # Mark the start of a header as a new sentence.
        for token in doc:
            if header_start <= token.idx:
                token.is_sent_start = True
                break
        # Mark the end of a header as a new sentence.
        for token in doc:
            if header_end <= token.idx:
                token.is_sent_start = True
                break
    return doc


def split_sentences(doc: str) -> list[str]:
    """Split a document into sentences."""

    def split_paragraphs(doc: str):
        """Split a document into paragraphs."""
        *paragraphs, last = doc.split("\n\n")
        return [paragraph + "\n\n" for paragraph in paragraphs] + ([last] if last else [])

    # Split sentences with spaCy.
    nlp = spacy.load("xx_sent_ud_sm")
    nlp.add_pipe("mark_additional_sentence_boundaries", before="senter")
    sentences = [sent.text_with_ws for sent in nlp(doc).sents if sent.text.strip()]
    # Split on paragraphs as well because SpaCy sentence splitting is not perfect.
    sentences = [paragraph for sent in sentences for paragraph in split_paragraphs(sent)]
    return sentences

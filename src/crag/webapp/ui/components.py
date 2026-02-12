from typing import Any

import streamlit as st

from crag.webapp.utils.common import build_filename_match_keys, extract_filename, format_metadata


def _is_ground_truth_match(ground_truth_keys: set[str] | None, *values: str) -> bool:
    if not ground_truth_keys:
        return False
    candidate_keys: set[str] = set()
    for value in values:
        candidate_keys.update(build_filename_match_keys(value))
    return bool(candidate_keys and candidate_keys.intersection(ground_truth_keys))


def _openai_result_match(
    filename: str, attributes: Any, ground_truth_keys: set[str] | None
) -> bool:
    candidates = [filename]
    if isinstance(attributes, dict):
        for field in ["url", "source", "filename", "file_name", "page_url"]:
            candidate = str(attributes.get(field, "") or "")
            if candidate:
                candidates.append(candidate)
    return _is_ground_truth_match(ground_truth_keys, *candidates)


def render_search_results(search_results: list[dict[str, Any]]) -> None:
    st.subheader("Search Results")
    for idx, result in enumerate(search_results, start=1):
        page_name = str(result.get("page_name", "") or "").strip()
        url = result.get("page_url", "")
        url_filename = extract_filename(str(url or ""))
        title = page_name or url_filename or f"Result {idx}"
        snippet = result.get("page_snippet", "")
        last_modified = result.get("page_last_modified", "")
        with st.expander(f"{idx}. {title}", expanded=False):
            if url:
                st.markdown(f"**URL:** [{url}]({url})")
            if last_modified:
                st.markdown(f"**Last Modified:** {last_modified}")
            if snippet:
                st.markdown("**Snippet:**")
                st.write(snippet)


def render_raglite(
    chunks: list[Any],
    answer: str,
    ground_truth_filename_keys: set[str] | None = None,
) -> None:
    st.subheader("RAGLite")
    st.markdown("**Answer:**")
    st.write(answer)
    st.markdown("**Chunks:**")
    for idx, chunk_span in enumerate(chunks, start=1):
        source = getattr(chunk_span, "document", None)
        url = getattr(source, "url", None) if source else None
        filename = getattr(source, "filename", None) if source else None
        display_name = (
            (filename or "").strip() or extract_filename(str(url or "")) or f"Chunk Span {idx}"
        )
        is_ground_truth_match = _is_ground_truth_match(
            ground_truth_filename_keys, (filename or "").strip(), str(url or "")
        )
        marker = "✅ " if is_ground_truth_match else "❌ "
        with st.expander(f"{idx}. {marker}{display_name}", expanded=False):
            if source and getattr(source, "metadata_", None):
                st.markdown("**Document Metadata:**")
                st.code(format_metadata(source.metadata_))
            for chunk_idx, chunk in enumerate(getattr(chunk_span, "chunks", []), start=1):
                st.markdown(f"**Chunk {chunk_idx}**")
                if getattr(chunk, "metadata_", None):
                    st.code(format_metadata(chunk.metadata_))
                headings = getattr(chunk, "headings", "").strip()
                body = getattr(chunk, "body", "").strip()
                if headings:
                    st.markdown("**Headings:**")
                    st.write(headings)
                if body:
                    st.markdown("**Body:**")
                    st.write(body)


def render_openai(
    results: Any,
    answer: str,
    citations: list[dict[str, Any]],
    tool_results: list[Any],
    ground_truth_filename_keys: set[str] | None = None,
) -> None:
    st.subheader("OpenAI")
    st.markdown("**Answer:**")
    st.write(answer)
    st.markdown("**Chunks:**")
    if results:
        for idx, result in enumerate(results, start=1):
            filename = str(getattr(result, "filename", "") or "").strip()
            display_name = filename or f"Result {idx}"
            attributes = getattr(result, "attributes", None)
            is_ground_truth_match = _openai_result_match(
                filename, attributes, ground_truth_filename_keys
            )
            marker = "✅ " if is_ground_truth_match else "❌ "
            title = f"{idx}. {marker}{display_name} (score: {result.score:.4f})"
            with st.expander(title, expanded=False):
                st.markdown(f"**File ID:** {result.file_id}")
                if attributes:
                    st.markdown("**Attributes:**")
                    st.code(format_metadata(attributes))
                for content_idx, content in enumerate(result.content, start=1):
                    st.markdown(f"**Content {content_idx}:**")
                    st.write(content.text)
    elif tool_results:
        for idx, result in enumerate(tool_results, start=1):
            filename = str(getattr(result, "filename", "") or "").strip()
            score = getattr(result, "score", None)
            attributes = getattr(result, "attributes", None)
            is_ground_truth_match = _openai_result_match(
                filename, attributes, ground_truth_filename_keys
            )
            marker = "✅ " if is_ground_truth_match else "❌ "
            title_bits = [str(idx), f"{marker}{filename or 'file_search result'}"]
            if score is not None:
                title_bits.append(f"score: {score:.4f}")
            title = " - ".join(title_bits)
            with st.expander(title, expanded=False):
                file_id = getattr(result, "file_id", None)
                if file_id:
                    st.markdown(f"**File ID:** {file_id}")
                if attributes:
                    st.markdown("**Attributes:**")
                    st.code(format_metadata(attributes))
                text = getattr(result, "text", None)
                if text:
                    st.markdown("**Text:**")
                    st.write(text)
    elif citations:
        for idx, citation in enumerate(citations, start=1):
            title = f"{idx}. {citation.get('type', 'citation')}"
            with st.expander(title, expanded=False):
                st.code(format_metadata(citation))
    else:
        st.info("No chunks or citations returned.")

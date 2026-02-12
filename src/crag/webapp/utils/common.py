import os
import re
from typing import Any
from urllib.parse import unquote, urlparse

_MISSING_PREDICTION_RE = re.compile(
    r"\b(i\s+don['’]?t\s+know|not\s+sure|cannot\s+determine|can['’]?t\s+determine|"
    r"insufficient\s+(info|information|context|evidence)|not\s+enough\s+(info|information|context|evidence)|"
    r"unknown|no\s+information|unable\s+to\s+answer|cannot\s+answer|can['’]?t\s+answer)\b",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SEPARATOR_RE = re.compile(r"[_\-/\\]+")
_TEXT_EXTENSIONS = {
    "md",
    "markdown",
    "txt",
    "text",
    "html",
    "htm",
    "pdf",
    "json",
    "csv",
    "xml",
}


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default


def format_metadata(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    lines: list[str] = []
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip()
    if not base_url:
        return ""
    if "/openai/v1" in base_url or base_url.endswith("/v1") or base_url.endswith("/v1/"):
        return base_url
    if base_url.endswith("/"):
        return f"{base_url}openai/v1/"
    return f"{base_url}/openai/v1/"


def build_judge_system_message(instructions: str, examples: str) -> str:
    instructions = (instructions or "").strip()
    examples = (examples or "").strip()
    if instructions and examples:
        return f"{instructions}\n\n{examples}"
    return instructions or examples


def is_missing_prediction(prediction: str) -> bool:
    cleaned = (prediction or "").strip()
    if not cleaned:
        return True
    return bool(_MISSING_PREDICTION_RE.search(cleaned))


def extract_filename(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""

    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        text = unquote(parsed.path or "")

    text = text.rstrip("/").replace("\\", "/")
    if not text:
        return ""
    return text.split("/")[-1].strip()


def normalize_filename_key(value: str) -> str:
    return extract_filename(value).lower()


def build_filename_match_keys(value: str) -> set[str]:
    filename = extract_filename(value).lower().strip()
    if not filename:
        return set()

    keys: set[str] = {filename}

    if "." in filename:
        stem, ext = filename.rsplit(".", 1)
        if ext in _TEXT_EXTENSIONS and stem:
            keys.add(stem)

    expanded: set[str] = set(keys)
    for key in keys:
        normalized = _SEPARATOR_RE.sub(" ", key)
        normalized = _WHITESPACE_RE.sub(" ", normalized).strip(" ._-")
        if normalized:
            expanded.add(normalized)

    for key in list(expanded):
        alnum = _NON_ALNUM_RE.sub("", key)
        if alnum:
            expanded.add(alnum)

    return {key for key in expanded if key}


def build_ground_truth_filename_keys(search_results: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for result in search_results:
        page_name = str(result.get("page_name", "") or "")
        page_url = str(result.get("page_url", "") or "")
        keys.update(build_filename_match_keys(page_name))
        keys.update(build_filename_match_keys(page_url))
    return keys

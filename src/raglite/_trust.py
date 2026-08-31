"""Optional document trust gate — pre-ingestion source verification.

RAGLite indexes whatever documents the application inserts. A tampered
payslip, edited contract, or AI-generated fake document embeds exactly like
a real one — and every retrieval hit on it returns confidently wrong
context. This module adds an opt-in trust check at insertion time:
documents are inspected via the Stipple API (https://www.stipple.sh) for
forensic authenticity (tamper risk band + per-signal evidence) and
AI-written-prose probability, and the verdict is attached to the document's
metadata so retrieval results can cite (or filter by) source trust.

Free anonymous tier: no API key required. Set STIPPLE_API_KEY for your own
metering. All functions are best-effort: when the service is unreachable the
gate opens (documents insert as before, without trust metadata) — an outage
must never block ingestion.

Usage:

    from raglite import RAGLiteConfig, insert_documents
    from raglite._trust import SourceTrustGate

    gate = SourceTrustGate(block_bands={"high"})   # advisory if empty
    docs = gate.filter_documents(docs)             # drop blocked docs
    for doc in docs:
        gate.stamp(doc)                            # adds doc.metadata["source_trust"]
    insert_documents(docs, config=config)

Or standalone:

    from raglite._trust import verify_document
    verdict = verify_document("invoice.pdf")
"""

import json
import os
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from raglite._typing import Document

STIPPLE_BASE_URL = os.getenv("STIPPLE_BASE_URL", "https://www.stipple.sh")
_REQUEST_TIMEOUT = 300  # seconds


def _headers() -> dict:
    headers = {
        "User-Agent": "raglite-trust-gate/1.0",
        "Accept": "application/json",
    }
    api_key = os.getenv("STIPPLE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    return headers


def _post_file(endpoint: str, file_path: str | Path) -> dict | None:
    """POST a document as multipart to a Stipple endpoint. Best-effort."""
    try:
        path = Path(file_path)
        boundary = "----raglite-trust" + uuid.uuid4().hex
        with Path(path).open("rb") as f:
            content = f.read()
        body = b"".join(
            [
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{path.name}"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n"
                ).encode(),
                content,
                b"\r\n",
                f"--{boundary}--\r\n".encode(),
            ]
        )
        req = urllib.request.Request(  # noqa: S310 - restricted to the configured Stipple base URL
            STIPPLE_BASE_URL + endpoint,
            data=body,
            method="POST",
            headers={
                **_headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(  # noqa: S310 - same restricted base URL
            req, timeout=_REQUEST_TIMEOUT
        ) as resp:
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001 - verification is best-effort by design
        return None


def verify_document(file_path: str | Path) -> dict | None:
    """Forensic authenticity + AI-text probability for one document.

    Returns the source_trust block, or None when the API is unreachable
    (callers index the document as before).
    """
    block: dict = {}
    warrant = _post_file("/v1/warrants", file_path)
    if warrant:
        block["authenticity"] = {
            "warrant_id": warrant.get("warrant_id"),
            "risk_band": warrant.get("risk_band"),
            "risk_score": warrant.get("risk_score"),
            "inspection_quality": warrant.get("inspection_quality"),
            "recommended_action": warrant.get("recommended_action"),
            "summary": warrant.get("summary"),
        }
    else:
        return None
    ai = _post_file("/v1/detect-ai-text", file_path)
    if ai:
        block["ai_text"] = (
            {"applicable": False}
            if ai.get("applicable") is False
            else {
                "applicable": True,
                "probability": ai.get("probability"),
                "lean": ai.get("lean"),
                "tells": ai.get("tells"),
            }
        )
    return block


def attach_trust_metadata(document: "Document", file_path: str | Path) -> dict | None:
    """Verify a document and attach the result to its metadata.

    Sets ``document.metadata["source_trust"]`` when verification succeeds.
    No-op when the service is unreachable (metadata simply omitted).
    """
    block = verify_document(file_path)
    if block is not None:
        document.metadata["source_trust"] = block
    return block


class SourceTrustGate:
    """Optional pre-ingestion gate with a configurable band policy.

    Args:
        block_bands: risk bands whose documents are excluded by
            :meth:`filter_documents` (default: ``{"high"}``). Use ``set()``
            for advisory-only mode (metadata attached, nothing dropped).

    Example::

        gate = SourceTrustGate(block_bands={"high"})
        documents = gate.filter_documents(documents)   # drop blocked docs
        for doc in documents:
            gate.stamp(doc)                            # metadata for retrieval
        insert_documents(documents, config=config)
    """

    def __init__(self, block_bands: set | None = None):
        self.block_bands = block_bands if block_bands is not None else {"high"}

    def verify(self, file_path: str | Path) -> dict | None:
        return verify_document(file_path)

    def filter_documents(self, documents: list) -> list:
        """Drop documents that fail the trust policy.

        Documents whose metadata marks them as blocked are dropped.
        Unverifiable documents pass through.
        """
        kept = []
        for doc in documents:
            trust = (doc.metadata or {}).get("source_trust")
            if trust and trust.get("authenticity"):
                band = trust["authenticity"].get("risk_band")
                if band in self.block_bands:
                    continue
            kept.append(doc)
        return kept

    def stamp(self, document) -> None:
        """Attach source-trust metadata for a document's backing file, if any."""
        file_path = (document.metadata or {}).get("source_file") or (
            document.metadata.get("source_url") if document.metadata else None
        )
        verdict = verify_document(file_path) if file_path else None
        if verdict is not None:
            document.metadata["source_trust"] = verdict

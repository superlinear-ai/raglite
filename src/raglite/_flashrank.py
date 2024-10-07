"""Patched version of FlashRankRanker that fixes incorrect reranking [1].

[1] https://github.com/AnswerDotAI/rerankers/issues/39
"""

import contextlib
from io import StringIO
from typing import Any

from flashrank import RerankRequest

# Suppress rerankers output on import until [1] is fixed.
# [1] https://github.com/AnswerDotAI/rerankers/issues/36
with contextlib.redirect_stdout(StringIO()):
    from rerankers.documents import Document
    from rerankers.models.flashrank_ranker import FlashRankRanker
    from rerankers.results import RankedResults, Result
    from rerankers.utils import prep_docs


class PatchedFlashRankRanker(FlashRankRanker):
    def rank(
        self,
        query: str,
        docs: str | list[str] | Document | list[Document],
        doc_ids: list[str] | list[int] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)
        passages = [{"id": doc_idx, "text": doc.text} for doc_idx, doc in enumerate(docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        flashrank_results = self.model.rerank(rerank_request)
        ranked_results = [
            Result(
                document=docs[result["id"]],  # This patches the incorrect ranking in the original.
                score=result["score"],
                rank=idx + 1,
            )
            for idx, result in enumerate(flashrank_results)
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

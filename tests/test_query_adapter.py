"""Test RAGLite's query adapter."""

from dataclasses import replace

import numpy as np
import pytest

from raglite import RAGLiteConfig, insert_evals, update_query_adapter, vector_search
from raglite._database import IndexMetadata


@pytest.mark.slow
def test_query_adapter(raglite_test_config: RAGLiteConfig) -> None:
    """Test the query adapter update functionality."""
    # Create a config with and without the query adapter enabled.
    config_with_query_adapter = replace(raglite_test_config, vector_search_query_adapter=True)
    config_without_query_adapter = replace(raglite_test_config, vector_search_query_adapter=False)
    # Verify that there is no query adapter in the database.
    Q = IndexMetadata.get("default", config=config_without_query_adapter).get("query_adapter")  # noqa: N806
    assert Q is None
    # Insert evals.
    insert_evals(num_evals=2, max_chunks_per_eval=10, config=config_with_query_adapter)
    # Update the query adapter.
    A = update_query_adapter(config=config_with_query_adapter)  # noqa: N806
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2  # noqa: PLR2004
    assert A.shape[0] == A.shape[1]
    assert np.isfinite(A).all()
    # Verify that there is a query adapter in the database.
    Q = IndexMetadata.get("default", config=config_without_query_adapter).get("query_adapter")  # noqa: N806
    assert isinstance(Q, np.ndarray)
    assert Q.ndim == 2  # noqa: PLR2004
    assert Q.shape[0] == Q.shape[1]
    assert np.isfinite(Q).all()
    assert np.all(A == Q)
    # Verify that the query adapter affects the results of vector search.
    query = "How does Einstein define 'simultaneous events' in his special relativity paper?"
    _, scores_qa = vector_search(query, config=config_with_query_adapter)
    _, scores_no_qa = vector_search(query, config=config_without_query_adapter)
    assert scores_qa != scores_no_qa

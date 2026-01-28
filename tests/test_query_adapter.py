"""Test RAGLite's query adapter."""

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from scipy.optimize import check_grad

from raglite import RAGLiteConfig, insert_evals, update_query_adapter, vector_search
from raglite._database import IndexMetadata
from raglite._query_adapter import _gradient, _objective_function


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
    insert_evals(num_evals=10, max_chunks_per_eval=10, config=config_with_query_adapter)
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


@pytest.mark.parametrize(
    "metric",
    [
        pytest.param("cosine", id="metric=cosine"),
        pytest.param("dot", id="metric=dot"),
    ],
)
@pytest.mark.parametrize(
    "embedding_dim",
    [
        pytest.param(16, id="embedding_dim=16"),
        pytest.param(128, id="embedding_dim=128"),
    ],
)
@pytest.mark.parametrize(
    "num_evals",
    [
        pytest.param(16, id="num_evals=16"),
        pytest.param(128, id="num_evals=128"),
    ],
)
def test_query_adapter_grad(
    num_evals: int, embedding_dim: int, metric: Literal["cosine", "dot"]
) -> None:
    """Verify that the query adapter gradient is correct."""
    # Generate test data.
    num_val = round(0.2 * num_evals)
    num_train = num_evals - num_val
    rng = np.random.default_rng(42)
    w0 = np.abs(rng.normal(size=num_train))
    Q_train = rng.normal(size=(num_train, embedding_dim))  # noqa: N806
    T_train = rng.normal(size=(num_train, embedding_dim))  # noqa: N806
    PT_train = rng.normal(size=(embedding_dim, embedding_dim))  # noqa: N806
    Q_val = rng.normal(size=(num_val, embedding_dim))  # noqa: N806
    D_val = rng.normal(size=(num_val, embedding_dim))  # noqa: N806
    config = RAGLiteConfig(vector_search_distance_metric=metric)
    # Check the gradient.
    l2_residual = check_grad(
        _objective_function, _gradient, w0, Q_train, T_train, PT_train, Q_val, D_val, config
    )
    assert (l2_residual / len(w0)) <= 100 * np.sqrt(np.finfo(w0.dtype).eps)

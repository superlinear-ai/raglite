"""Test RAGLite."""

import raglite


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(raglite.__name__, str)

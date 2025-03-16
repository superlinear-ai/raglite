"""Test that llama-cpp-python package is an optional dependency for RAGLite."""

import builtins
import sys
from typing import Any

import pytest


def test_raglite_import_without_llama_cpp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that RAGLite can be imported without llama-cpp-python being available."""
    # Unimport raglite and llama_cpp.
    module_names = list(sys.modules)
    for module_name in module_names:
        if module_name.startswith(("llama_cpp", "raglite", "sqlmodel")):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    # Save the original __import__ function.
    original_import = builtins.__import__

    # Define a fake import function that raises ModuleNotFoundError when trying to import llama_cpp.
    def fake_import(name: str, *args: Any) -> Any:
        if name.startswith("llama_cpp"):
            import_error = f"No module named '{name}'"
            raise ModuleNotFoundError(import_error)
        return original_import(name, *args)

    # Monkey patch __import__ with the fake import function.
    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Verify that importing raglite does not raise an error.
    import raglite  # noqa: F401
    from raglite._config import llama_supports_gpu_offload  # type: ignore[attr-defined]

    # Verify that lazily using llama-cpp-python raises a ModuleNotFoundError.
    with pytest.raises(ModuleNotFoundError, match="llama.cpp models"):
        llama_supports_gpu_offload()

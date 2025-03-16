"""Import from llama-cpp-python with a lazy ModuleNotFoundError if it's not installed."""

from importlib import import_module
from typing import TYPE_CHECKING, Any, NoReturn

# When type checking, import everything normally.
if TYPE_CHECKING:
    from llama_cpp import (  # type: ignore[attr-defined]
        LLAMA_POOLING_TYPE_NONE,
        Llama,
        LlamaRAMCache,
        llama,
        llama_chat_format,
        llama_grammar,
        llama_supports_gpu_offload,
        llama_types,
    )

# Explicitly export these names for static analysis.
__all__ = [
    "LLAMA_POOLING_TYPE_NONE",
    "Llama",
    "LlamaRAMCache",
    "llama",
    "llama_chat_format",
    "llama_grammar",
    "llama_supports_gpu_offload",
    "llama_types",
]


def __getattr__(name: str) -> object:
    """Import from llama-cpp-python with a lazy ModuleNotFoundError if it's not installed."""

    # Create a mock attribute and submodule that lazily raises an ModuleNotFoundError when accessed.
    class LazyAttributeError:
        error_message = "To use llama.cpp models, please install `llama-cpp-python`."

        def __init__(self, error: ModuleNotFoundError | None = None):
            self.error = error

        def __getattr__(self, name: str) -> NoReturn:
            raise ModuleNotFoundError(self.error_message) from self.error

        def __call__(self, *args: Any, **kwargs: Any) -> NoReturn:
            raise ModuleNotFoundError(self.error_message) from self.error

    class LazySubmoduleError:
        def __init__(self, error: ModuleNotFoundError):
            self.error = error

        def __getattr__(self, name: str) -> LazyAttributeError | type[LazyAttributeError]:
            return LazyAttributeError(self.error) if name == name.lower() else LazyAttributeError

    # Check if the attribute is a submodule.
    llama_cpp_submodules = ["llama", "llama_chat_format", "llama_grammar", "llama_types"]
    attr_is_submodule = name in llama_cpp_submodules
    try:
        # Import and return the requested submodule or attribute.
        module = import_module(f"llama_cpp.{name}" if attr_is_submodule else "llama_cpp")
        return module if attr_is_submodule else getattr(module, name)
    except ModuleNotFoundError as import_error:
        # Return a mock submodule or attribute that lazily raises an ModuleNotFoundError.
        return (
            LazySubmoduleError(import_error)
            if attr_is_submodule
            else LazyAttributeError(import_error)
        )

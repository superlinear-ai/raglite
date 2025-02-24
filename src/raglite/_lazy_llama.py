"""Import llama_cpp lazily to avoid import errors when it is not installed."""

from importlib import import_module
from typing import TYPE_CHECKING

# When type checking, import everything normally.
if TYPE_CHECKING:
    from llama_cpp import (  # type: ignore[attr-defined]
        LLAMA_POOLING_TYPE_NONE,
        Llama,
        LlamaRAMCache,
        llama,
        llama_grammar,
        llama_supports_gpu_offload,
        llama_types,
    )
    from llama_cpp.llama_chat_format import (
        _convert_completion_to_chat,
        _convert_completion_to_chat_function,
        _grammar_for_response_format,
    )
    from llama_cpp.llama_types import (
        ChatCompletionRequestMessage,
        ChatCompletionTool,
        ChatCompletionToolChoiceOption,
        CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse,
    )

# Explicitly export these names for static analysis.
__all__ = [
    "llama",
    "llama_grammar",
    "llama_types",
    "Llama",
    "LLAMA_POOLING_TYPE_NONE",
    "llama_supports_gpu_offload",
    "LlamaRAMCache",
    "_convert_completion_to_chat",
    "_convert_completion_to_chat_function",
    "_grammar_for_response_format",
    "ChatCompletionRequestMessage",
    "ChatCompletionTool",
    "ChatCompletionToolChoiceOption",
    "CreateChatCompletionResponse",
    "CreateChatCompletionStreamResponse",
]


# Module names for the submodules of llama_cpp.
LLAMA_CPP_MODULE_NAME = "llama_cpp"
CHAT_SUBMODULE_NAME = "llama_chat_format"
TYPES_SUBMODULE_NAME = "llama_types"

# Map attributes that live in submodules to their module names.
_SUBMODULE_ATTRS = {
    # Attributes from llama_cpp.llama_chat_format
    "_convert_completion_to_chat": f"{LLAMA_CPP_MODULE_NAME}.{CHAT_SUBMODULE_NAME}",
    "_convert_completion_to_chat_function": f"{LLAMA_CPP_MODULE_NAME}.{CHAT_SUBMODULE_NAME}",
    "_grammar_for_response_format": f"{LLAMA_CPP_MODULE_NAME}.{CHAT_SUBMODULE_NAME}",
    # Attributes from llama_cpp.llama_types
    "ChatCompletionRequestMessage": f"{LLAMA_CPP_MODULE_NAME}.{TYPES_SUBMODULE_NAME}",
    "ChatCompletionTool": f"{LLAMA_CPP_MODULE_NAME}.{TYPES_SUBMODULE_NAME}",
    "ChatCompletionToolChoiceOption": f"{LLAMA_CPP_MODULE_NAME}.{TYPES_SUBMODULE_NAME}",
    "CreateChatCompletionResponse": f"{LLAMA_CPP_MODULE_NAME}.{TYPES_SUBMODULE_NAME}",
    "CreateChatCompletionStreamResponse": f"{LLAMA_CPP_MODULE_NAME}.{TYPES_SUBMODULE_NAME}",
    # Attributes from the top-level llama_cpp module.
    "llama": LLAMA_CPP_MODULE_NAME,
    "llama_grammar": LLAMA_CPP_MODULE_NAME,
    "llama_types": LLAMA_CPP_MODULE_NAME,
    "Llama": LLAMA_CPP_MODULE_NAME,
    "LLAMA_POOLING_TYPE_NONE": LLAMA_CPP_MODULE_NAME,
}


def __getattr__(name: str) -> object:
    """Import the requested attribute from the llama_cpp module lazily."""
    module_name = _SUBMODULE_ATTRS.get(name, LLAMA_CPP_MODULE_NAME)

    try:
        module = import_module(module_name)
    except ImportError as e:
        import_error_message = (
            "llama-cpp-python is required for local language model support.\n"
            "Install it with `pip install raglite[llama-cpp-python]`."
        )
        raise ImportError(import_error_message) from e

    try:
        return getattr(module, name)
    except AttributeError as e:
        attribute_error_message = f"Module '{module_name}' has no attribute '{name}'"
        raise AttributeError(attribute_error_message) from e

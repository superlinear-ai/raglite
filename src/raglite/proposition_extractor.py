"""Extract propositions from a chunk."""

import json
import warnings
from collections.abc import Callable

from llama_cpp import Llama

from raglite.llm import default_llm

RESPONSE_SCHEMA = {
    "type": "object",
    "description": "A MECE list of questions about a document chunk.",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A single question about the document chunk.",
        }
    },
    "required": ["questions"],
}

SYSTEM_PROMPT = """
You are a subject matter expert on the topic "{document_topic}".
You are given the headers and body of a chunk from a document.
Your task is to write a MECE list of questions to quiz other subject matter experts on the information in the chunk's body.
ONLY include questions that are answered completely by the chunk's body.
ALWAYS ensure each question is self-contained and explicitly restates any subjects or objects it refers to.
NEVER write redundant qualifiers such as "according to the text", "according to the authors", or "in the document".
ALWAYS format your response according to this JSON schema:
```
{response_schema}
```
""".strip()

USER_PROMPT = """
Chunk headers:
```
{chunk_headers}
```

Chunk body:
```
{chunk_body}
```
""".strip()


def extract_propositions(  # noqa: PLR0913
    document_topic: str,
    chunk_headers: str,
    chunk_body: str,
    max_tries: int = 4,
    temperature: float = 0.7,
    llm: Callable[[], Llama] = default_llm,
) -> list[str]:
    """Extract propositions from a chunk."""
    system_prompt = SYSTEM_PROMPT.format(
        document_topic=document_topic,
        response_schema=json.dumps(RESPONSE_SCHEMA, indent=None),
    )
    user_prompt = USER_PROMPT.format(
        chunk_headers=chunk_headers.strip(), chunk_body=chunk_body.strip()
    )
    propositions = []
    for _ in range(max_tries):
        response = llm().create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object", "schema": RESPONSE_SCHEMA},
            temperature=temperature,
        )
        try:
            # Parse the propositions from the response.
            propositions = json.loads(response["choices"][0]["message"]["content"])["questions"]
            # Basic quality checks.
            if len(chunk_body) >= 42 and not propositions:  # noqa: PLR2004
                raise ValueError  # noqa: TRY301
            if propositions and not all(proposition[0].isupper() for proposition in propositions):
                raise ValueError  # noqa: TRY301
        except Exception:  # noqa: S112, BLE001
            continue
        else:
            break
    else:
        warnings.warn(
            f"Failed to extract propositions from:\n\n{chunk_headers}\n\n{chunk_body}", stacklevel=2
        )
    return propositions

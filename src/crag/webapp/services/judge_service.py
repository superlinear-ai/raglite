import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from crag.webapp.utils.common import normalize_base_url

ABSTAIN_RE = re.compile(
    r"\b(i\s+don['’]?t\s+know|not\s+sure|cannot\s+determine|can['’]?t\s+determine|"
    r"insufficient\s+(info|information|context|evidence)|not\s+enough\s+(info|information|context|evidence)|"
    r"unknown|no\s+information|unable\s+to\s+answer|cannot\s+answer|can['’]?t\s+answer)\b",
    re.IGNORECASE,
)


class CRAGLabel(str, Enum):
    correct = "correct"
    incorrect = "incorrect"
    missing = "missing"


class CRAGResponse(BaseModel):
    explanation: str = Field(..., max_length=300)
    label: CRAGLabel = Field(...)


def run_judge_once(
    *,
    query: str,
    ground_truth: str,
    prediction: str,
    system_message: str,
    model_name: str,
    base_url: str,
    api_key: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    try:
        from openai import APIConnectionError, OpenAI, RateLimitError
    except Exception as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc

    client = OpenAI(base_url=normalize_base_url(base_url) or None, api_key=api_key)
    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": (
                f"Question: {query}\n"
                f"Ground truth: {ground_truth}\n"
                f"Prediction: {prediction}\n"
            ),
        },
    ]

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            response = client.responses.parse(
                model=model_name,
                input=messages,
                text_format=CRAGResponse,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Judge returned no parsed response.")
            return {"label": parsed.label.value, "explanation": parsed.explanation}
        except (APIConnectionError, RateLimitError) as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            break

    if last_error:
        raise last_error
    raise RuntimeError("Judge failed without a specific error.")

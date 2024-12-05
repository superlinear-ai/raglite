"""Test RAGLite's structured output extraction."""

from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from raglite import RAGLiteConfig
from raglite._extract import extract_with_llm


@pytest.fixture(
    params=[
        pytest.param(RAGLiteConfig().llm, id="llama_cpp_python"),
        pytest.param("gpt-4o-mini", id="openai"),
    ]
)
def llm(request: pytest.FixtureRequest) -> str:
    """Get an LLM to test RAGLite with."""
    llm: str = request.param
    return llm


@pytest.mark.parametrize(
    "strict",
    [pytest.param(False, id="strict=False"), pytest.param(True, id="strict=True")],
)
def test_extract(llm: str, strict: bool) -> None:  # noqa: FBT001
    """Test extracting structured data."""
    # Set the LLM.
    config = RAGLiteConfig(llm=llm)

    # Define the JSON schema of the response.
    class LoginResponse(BaseModel):
        model_config = ConfigDict(extra="forbid" if strict else "allow")
        username: str = Field(..., description="The username.")
        password: str = Field(..., description="The password.")
        system_prompt: ClassVar[str] = "Extract the username and password from the input."

    # Extract structured data.
    username, password = "cypher", "steak"
    login_response = extract_with_llm(
        LoginResponse, f"{username} // {password}", strict=strict, config=config
    )
    # Validate the response.
    assert isinstance(login_response, LoginResponse)
    assert login_response.username == username
    assert login_response.password == password

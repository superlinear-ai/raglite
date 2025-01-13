"""Test RAGLite's structured output extraction."""

from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, Field

from raglite import RAGLiteConfig
from raglite._extract import extract_with_llm


@pytest.mark.parametrize(
    "strict", [pytest.param(False, id="strict=False"), pytest.param(True, id="strict=True")]
)
def test_extract(llm: str, strict: bool) -> None:  # noqa: FBT001
    """Test extracting structured data."""
    # Set the LLM.
    config = RAGLiteConfig(llm=llm)

    # Define the JSON schema of the response.
    class UserProfileResponse(BaseModel):
        """The response to a user profile extraction request."""

        model_config = ConfigDict(extra="forbid" if strict else "allow")
        username: str = Field(..., description="The username.")
        email: str = Field(..., description="The email address.")
        system_prompt: ClassVar[str] = "Extract the username and email from the input."

    # Example input data.
    username, email = "cypher", "cypher@example.com"
    profile_response = extract_with_llm(
        UserProfileResponse, f"username: {username}\nemail: {email}", strict=strict, config=config
    )
    # Validate the response.
    assert isinstance(profile_response, UserProfileResponse)
    assert profile_response.username == username
    assert profile_response.email == email

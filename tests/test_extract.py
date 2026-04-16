"""Test RAGLite's structured output extraction."""

from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from raglite import Document, RAGLiteConfig
from raglite._extract import expand_document_metadata, extract_with_llm


@pytest.mark.parametrize(
    "strict", [pytest.param(False, id="strict=False"), pytest.param(True, id="strict=True")]
)
def test_extract(llm: str, strict: bool) -> None:  # noqa: FBT001
    """Test extracting structured data."""
    # Set the LLM.
    config = RAGLiteConfig(llm=llm)

    # Define the JSON schema of the response.
    class LoginResponse(BaseModel):
        """The response to a login request."""

        model_config = ConfigDict(extra="forbid" if strict else "allow")
        username: str = Field(..., description="The username.")
        password: str = Field(..., description="The password.")
        system_prompt: ClassVar[str] = "Extract the username and password from the input."

    # Extract structured data.
    username, password = "cypher", "steak"
    login_response = extract_with_llm(
        LoginResponse, f"username: {username}\npassword: {password}", strict=strict, config=config
    )
    # Validate the response.
    assert isinstance(login_response, LoginResponse)
    assert login_response.username == username
    assert login_response.password == password


def test_expand_document_metadata(llm: str) -> None:
    """Supports all metadata field types and preserves existing metadata, including optional."""
    config = RAGLiteConfig(llm=llm)
    document = Document(
        id="mission",
        filename="mission.txt",
        url=None,
        metadata_={"summary": "A concise overview of the Mars mission."},
        content=(
            "Title: The Mars Mission Manual. The reference guide runs 42 pages and summarizes "
            "planetary geology from recent expeditions. Seasoned explorers Alice and Bob co-authored "
            "the manual after a decade of joint work. Reviewers rate the collection 4.5 out of 5 "
            "stars and classify it under the 'Planet' category. Key topics include Exploration and "
            "Geology, with dedicated sections that walk through reconnaissance protocols and rock "
            "sampling strategies. Chief editor Dr. Elena Martinez coordinated the publication."
        ),
    )
    metadata_fields = {
        "title": Annotated[str, Field(..., description="Document title.")],
        "pages": Annotated[int, Field(..., description="Total page count.")],
        "rating": Annotated[float, Field(..., description="Average review score.")],
        "reviewed": Annotated[bool, Field(..., description="Whether peer reviewed.")],
        "category": Annotated[
            Literal["Planet", "Moon"],
            Field(..., description="Primary classification."),
        ],
        "tags": Annotated[
            list[Literal["Exploration", "Geology", "Biology"]],
            Field(..., description="Relevant themes."),
        ],
        "participants": Annotated[list[str], Field(..., description="Contributors.")],
        "editor": Annotated[
            str | None,
            Field(default=None, description="Editor name (if available)"),
        ],
    }
    enriched_document = next(
        expand_document_metadata([document], metadata_fields, config=config)  # type: ignore[arg-type]
    )
    # Basic checks (avoid overly brittle assertions across LLMs)
    assert enriched_document.metadata_["title"]
    assert enriched_document.metadata_["pages"] >= 1
    max_rating = 5.0
    assert 0.0 <= enriched_document.metadata_["rating"] <= max_rating
    assert enriched_document.metadata_["reviewed"] in (True, False)
    assert enriched_document.metadata_["category"] in {"Planet", "Moon"}
    assert "Exploration" in enriched_document.metadata_["tags"]
    assert set(enriched_document.metadata_["participants"]) >= {"Alice", "Bob"}
    # Optional field should be present when available
    assert enriched_document.metadata_["editor"] in ("Dr. Elena Martinez", None)
    assert enriched_document.metadata_["summary"] == "A concise overview of the Mars mission."

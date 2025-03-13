"""Test RAGLite's reranking against a golden dataset."""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pytest
from litellm import completion
from pytest import skip
from rerankers.models.flashrank_ranker import FlashRankRanker
from rerankers.models.ranker import BaseRanker
from scipy.stats import kendalltau

from raglite import RAGLiteConfig, hybrid_search, rerank_chunks, retrieve_chunks
from raglite._database import Chunk

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

T = TypeVar("T")


def kendall_tau(a: list[T], b: list[T]) -> float:
    """Measure the Kendall rank correlation coefficient between two lists."""
    try:
        τ: float = kendalltau(range(len(a)), [a.index(el) for el in b])[0]  # noqa: PLC2401
        return τ
    except Exception as e:
        logger.warning(f"Error calculating Kendall's Tau: {e}")
        return float("nan")


def load_golden_dataset() -> dict:
    """Load the golden dataset from a JSON file."""
    dataset_path = Path(__file__).parent / "data" / "golden_rankings.json"
    if not dataset_path.exists():
        # If the dataset doesn't exist, create it
        create_golden_dataset(dataset_path)

    with open(dataset_path) as f:
        return json.load(f)


def create_golden_dataset(output_path: Path) -> None:
    """Create a golden dataset using a powerful LLM.

    This creates an initial empty dataset structure that will be populated
    with actual chunk IDs and rankings when the test runs.

    The dataset structure is organized by model, then by query, allowing
    for caching rankings from different models.
    """
    os.makedirs(output_path.parent, exist_ok=True)

    # Create an empty dataset structure that will be populated later
    # Organized by model -> query -> rankings
    dataset = {
        "models": {
            "gpt-4o-mini": {
                "queries": {
                    "What does it mean for two events to be simultaneous?": {
                        # These IDs will be populated with actual chunk IDs when the test runs
                        "chunk_ids": [],
                        "metadata": {
                            "source": "litellm_completion",
                            "timestamp": datetime.now().isoformat(),
                            "prompt": "Rank these chunks by relevance to the query using litellm completion",
                            "temperature": 0.0,
                            "response_format": "json_object",
                        },
                    }
                }
            },
            "o3-mini": {
                "queries": {
                    "What does it mean for two events to be simultaneous?": {
                        # These IDs will be populated with actual chunk IDs when the test runs
                        "chunk_ids": [],
                        "metadata": {
                            "source": "litellm_completion",
                            "timestamp": datetime.now().isoformat(),
                            "prompt": "Rank these chunks by relevance to the query using litellm completion",
                            "temperature": 0.0,
                            "response_format": "json_object",
                        },
                    }
                }
            },
        }
    }

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)


def reset_golden_dataset() -> None:
    """Reset the golden dataset by removing the file."""
    dataset_path = Path(__file__).parent / "data" / "golden_rankings.json"
    if dataset_path.exists():
        os.remove(dataset_path)
        logger.info(f"Removed existing golden dataset at {dataset_path}")


def update_golden_dataset_with_real_chunks(
    query: str, chunks: list[Chunk], config: RAGLiteConfig, model_override: str = None
) -> list[str]:
    """Update the golden dataset with real chunk IDs.

    This function uses the LLM from the config to rank the chunks by relevance to the query.

    Args:
        query: The query to rank chunks for
        chunks: The chunks to rank
        config: The RAGLiteConfig
        model_override: Optional model to use instead of the one in config

    Returns
    -------
        The list of chunk IDs in ranked order
    """
    if not chunks:
        logger.warning("No chunks provided to update_golden_dataset_with_real_chunks")
        return []

    dataset_path = Path(__file__).parent / "data" / "golden_rankings.json"
    db_type = "postgres" if "postgres" in config.db_url else "sqlite"

    # Determine which model to use
    model_to_use = model_override or config.llm
    logger.info(f"Using model: {model_to_use} for ranking")

    # Load existing dataset
    if dataset_path.exists():
        with open(dataset_path) as f:
            try:
                dataset = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON from {dataset_path}, creating new dataset")
                dataset = {"models": {}}
    else:
        dataset = {"models": {}}

    # Ensure the dataset has the correct structure
    if "models" not in dataset:
        # Convert old format to new format if needed
        if "queries" in dataset:
            logger.info("Converting old dataset format to new format")
            old_queries = dataset["queries"]
            dataset = {"models": {"gpt-4o-mini": {"queries": old_queries}}}
        else:
            dataset = {"models": {}}

    # Ensure the model exists in the dataset
    if model_to_use not in dataset["models"]:
        dataset["models"][model_to_use] = {"queries": {}}

    if "queries" not in dataset["models"][model_to_use]:
        dataset["models"][model_to_use]["queries"] = {}

    # Create a unique key for this query and database type
    query_key = f"{query}_{db_type}"

    # Check if we already have a ranking for this query, database type, and model
    if (
        query_key in dataset["models"][model_to_use]["queries"]
        and dataset["models"][model_to_use]["queries"][query_key]["chunk_ids"]
    ):
        # Check if the cached chunks are stale
        golden_data = dataset["models"][model_to_use]["queries"][query_key]
        golden_chunk_ids = golden_data["chunk_ids"]
        current_chunk_ids = {chunk.id for chunk in chunks}

        # Check when the ranking was last updated
        last_updated = datetime.fromisoformat(golden_data["metadata"]["timestamp"])
        current_time = datetime.now()
        age_in_days = (current_time - last_updated).days

        logger.info(
            f"Found existing golden ranking for {model_to_use}/{query_key} with {len(golden_chunk_ids)} chunk IDs "
            f"(last updated {age_in_days} days ago)"
        )
        logger.info(f"Current chunks have {len(current_chunk_ids)} unique IDs")

        # Check overlap
        overlap = set(golden_chunk_ids).intersection(current_chunk_ids)
        overlap_percentage = len(overlap) / len(golden_chunk_ids) if golden_chunk_ids else 0
        logger.info(
            f"Overlap between golden and current: {len(overlap)} chunk IDs ({overlap_percentage:.1%})"
        )

        # Use existing ranking if:
        # 1. It's less than 30 days old, and
        # 2. There's at least 80% overlap with current chunks
        if age_in_days < 30 and overlap_percentage >= 0.8:
            logger.info(
                f"Using existing golden ranking for {model_to_use}/{query_key} (overlap: {overlap_percentage:.1%})"
            )
            return golden_chunk_ids

        if age_in_days >= 30:
            logger.info(f"Cached ranking is stale ({age_in_days} days old), creating new ranking")
        else:
            logger.info(f"Insufficient overlap ({overlap_percentage:.1%}), creating new ranking")

    # Log the chunks we're working with
    logger.info(f"Creating golden ranking for {model_to_use}/{query_key} with {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:5]):
        logger.info(f"  Chunk #{i}: ID={chunk.id[:8]}, Index={chunk.index}")
        logger.info(f"    Content (truncated): {chunk.body[:100]}...")

    # Use litellm's completion to rank the chunks
    # Prepare the chunks text for the prompt
    chunks_text = "\n\n".join(
        [f"Chunk {i} (ID: {chunk.id}):\n{chunk.body}" for i, chunk in enumerate(chunks)]
    )

    # Prepare the system and user messages for the LLM
    system_message = """You are an expert at evaluating the relevance of text chunks to a query. 
Your task is to rank chunks in order of relevance to the given query, with the most relevant first.

Take your time to reason carefully about each chunk before making your final decision. You can think step by step to ensure accurate evaluation.

For each chunk:
    1. Analyze its content in relation to the query.
    2. Provide a chain of thought explaining your reasoning.
    3. Assign a relevancy score from 0 to 10, where 10 is most relevant.

Put your reasoning in XML tags for each chunk. For example: 

<reasoning>
<chunk id="chunk_id_1">Your reasoning for this chunk. Final score: 8</chunk>
<chunk id="chunk_id_2">Your reasoning for this chunk. Final score: 10</chunk>
...and so on for all chunks...
</reasoning>

After evaluating all chunks, return your final ranking using XML tags as follows:
<ranking>
<chunk id="chunk_id_2" score="10">Your reasoning for this chunk.</chunk>
<chunk id="chunk_id_1" score="8">Your reasoning for this chunk.</chunk>
...and so on for all chunks...
</ranking>

The chunk IDs are unique identifiers like "d8281ef4" or "09840c90".
Do not simplify or modify the IDs in any way - use them exactly as provided."""

    user_message = f"""Query: "{query}"

Below are chunks from a document. Evaluate each chunk's relevance to the query:

{chunks_text}

For each chunk:
1. Analyze its content in relation to the query.
2. Provide your reasoning.
3. Assign a relevancy score from 0 to 10 (10 = most relevant).

You can reason through your evaluation process before finalizing your rankings. Take your time to consider:
- How directly the chunk answers the query
- The specificity and relevance of information in the chunk
- Whether the chunk contains key concepts related to the query
- How comprehensive the chunk's coverage of the query topic is

First, provide your detailed reasoning for each chunk using XML tags:
<reasoning>
<chunk id="[exact chunk ID]">Your reasoning for this chunk. Final score: [0-10]</chunk>
<chunk id="[exact chunk ID]">Your reasoning for this chunk. Final score: [0-10]</chunk>
...continue for all chunks...
</reasoning>

Then, provide your final ranking using XML tags, with chunks ordered from most to least relevant:
<ranking>
<chunk id="[exact chunk ID]" score="[0-10]">Brief summary of why this chunk is relevant.</chunk>
<chunk id="[exact chunk ID]" score="[0-10]">Brief summary of why this chunk is relevant.</chunk>
...continue for all chunks...
</ranking>
"""

    # Call the LLM to get the ranking
    logger.info("Calling LLM to rank chunks...")
    try:
        # Use the model specified (either from config or override)
        # Both gpt-4o-mini and o3-mini have large context windows
        # Only handle context window limits for local models (not gpt-4o-mini or o3-mini)
        is_local_model = "llama" in model_to_use.lower()

        # For gpt-4o-mini and o3-mini, we don't need to worry about context window
        if model_to_use in ["gpt-4o-mini", "o3-mini"]:
            is_local_model = False

        # Handle context window limits for local models
        if is_local_model:
            # Estimate tokens: ~1.3 tokens per word, ~4 chars per word -> ~0.33 tokens per char
            estimated_tokens = int(len(system_message + user_message) * 0.33)
            logger.info(f"Estimated token count for prompt: {estimated_tokens}")

            if estimated_tokens > 4000:  # Local models typically have 4096 context
                logger.warning("Prompt may exceed local model context window. Truncating chunks...")
                # Truncate chunks to fit context
                max_chars = int(4000 / (len(chunks) * 0.33))
                chunks_text = "\n\n".join(
                    [
                        f"Chunk {i} (ID: {chunk.id}):\n{chunk.body[:max_chars]}"
                        for i, chunk in enumerate(chunks)
                    ]
                )
                # Rebuild user message with truncated chunks
                user_message = f"""Query: "{query}"

Below are chunks from a document. Rank these chunks in order of relevance to the query, with the most relevant first.

{chunks_text}

Provide your ranking as a JSON array of chunk IDs, from most to least relevant.
Only return the JSON array, nothing else."""
                logger.info("Truncated prompt to approximately 4000 tokens")

        # Prepare completion parameters
        completion_params = {
            "model": model_to_use,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            # No response_format parameter since we're using XML tags instead of JSON
        }

        # Handle model-specific parameters
        if model_to_use == "o3-mini":
            # o3-mini doesn't support temperature=0.0, only temperature=1 is supported
            logger.info("Using o3-mini model, setting temperature=1.0")
            # No temperature parameter for o3-mini (it defaults to 1.0)
        else:
            # For other models, use temperature=0.0 for deterministic output
            completion_params["temperature"] = 0.0

        # Call the LLM
        response = completion(**completion_params)

        # Extract the response content
        llm_response = response.choices[0].message.content
        logger.info(f"LLM response: {llm_response}")

        # Parse the XML response
        try:
            # Extract both reasoning and ranking XML sections
            reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", llm_response, re.DOTALL)
            ranking_match = re.search(r"<ranking>(.*?)</ranking>", llm_response, re.DOTALL)

            if not ranking_match:
                raise ValueError("Could not find <ranking> tags in LLM response")

            # Process detailed reasoning if available
            detailed_reasoning = {}
            if reasoning_match:
                reasoning_xml = reasoning_match.group(1)
                # Extract chunk IDs and reasoning using regex
                reasoning_pattern = re.compile(r'<chunk id="([^"]+)">(.*?)</chunk>', re.DOTALL)
                reasoning_matches = reasoning_pattern.findall(reasoning_xml)

                # Create a mapping of chunk ID to reasoning
                for chunk_id, reasoning in reasoning_matches:
                    detailed_reasoning[chunk_id] = reasoning.strip()

                logger.info(f"Extracted detailed reasoning for {len(detailed_reasoning)} chunks")

            # Process the ranking
            ranking_xml = ranking_match.group(1)

            # Extract chunk IDs and scores using regex
            chunk_pattern = re.compile(
                r'<chunk id="([^"]+)" score="([^"]+)">(.*?)</chunk>', re.DOTALL
            )
            chunk_matches = chunk_pattern.findall(ranking_xml)

            # Create a list of (chunk_id, score, summary, reasoning) tuples
            chunk_evaluations = [
                (
                    chunk_id,
                    float(score),
                    summary.strip(),
                    detailed_reasoning.get(chunk_id, "No detailed reasoning provided"),
                )
                for chunk_id, score, summary in chunk_matches
            ]

            # Sort by score in descending order (if not already sorted)
            chunk_evaluations.sort(key=lambda x: x[1], reverse=True)

            # Extract just the chunk IDs in ranked order
            ranked_chunk_ids = [chunk_id for chunk_id, _, _, _ in chunk_evaluations]

            # Log the evaluations for the top chunks
            logger.info("Top chunk evaluations:")
            for i, (chunk_id, score, summary, reasoning) in enumerate(chunk_evaluations[:3]):
                logger.info(f"  #{i + 1}: ID={chunk_id[:8]}, Score={score}")
                logger.info(f"    Summary: {summary[:100]}...")
                logger.info(f"    Detailed reasoning: {reasoning[:150]}...")

            logger.info(f"Successfully parsed LLM ranking with {len(ranked_chunk_ids)} chunks")

            # Verify that all returned IDs are valid
            valid_ids = {chunk.id for chunk in chunks}
            invalid_ids = [id for id in ranked_chunk_ids if id not in valid_ids]
            if invalid_ids:
                logger.warning(f"LLM returned invalid chunk IDs: {invalid_ids}")
                ranked_chunk_ids = [id for id in ranked_chunk_ids if id in valid_ids]

            # Check if we have all the chunks
            missing_ids = valid_ids - set(ranked_chunk_ids)
            if missing_ids:
                logger.warning(f"LLM ranking is missing {len(missing_ids)} chunk IDs")
                # Append missing IDs to the end
                ranked_chunk_ids.extend(missing_ids)

            # Use the LLM ranking
            chunk_ids = ranked_chunk_ids
            logger.info(f"Using LLM ranking with {len(chunk_ids)} chunks")
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Fall back to default ranking
            chunk_ids = [chunk.id for chunk in chunks]
    except Exception as e:
        logger.warning(f"Error calling LLM: {e}")
        # Fall back to default ranking
        chunk_ids = [chunk.id for chunk in chunks]

    # Log the ranked chunks
    logger.info("LLM-ranked chunks (first 5):")
    for i, chunk_id in enumerate(chunk_ids[:5]):
        # Find the chunk with this ID
        chunk = next((c for c in chunks if c.id == chunk_id), None)
        if chunk:
            logger.info(f"  #{i}: ID={chunk_id[:8]}, Index={chunk.index}")
            logger.info(f"    Content (truncated): {chunk.body[:100]}...")

    # Update the dataset
    # Create metadata with information about the model
    metadata = {
        "source": "litellm_completion",
        "timestamp": datetime.now().isoformat(),
        "prompt": f"Ranked by LLM relevance using litellm completion with {model_to_use}",
        "db_type": db_type,
        "llm": model_to_use,
        "temperature": 0.0,
        "format": "xml_ranking_with_reasoning",
        "scoring": "0-10 relevance scale with step-by-step reasoning",
    }

    # Update or create the entry for this query
    dataset["models"][model_to_use]["queries"][query_key] = {
        "chunk_ids": chunk_ids,
        "metadata": metadata,
        "evaluations": [
            {
                "chunk_id": chunk_id,
                "score": float(score),
                "summary": summary,
                "reasoning": reasoning,
                "index": next((c.index for c in chunks if c.id == chunk_id), None),
                "body": re.sub(r"\s+", " ", next((c.body for c in chunks if c.id == chunk_id), ""))[
                    :600
                ]
                + "..."
                if next((c.body for c in chunks if c.id == chunk_id), "")
                else "",
                "chunk_len": len(next((c.body for c in chunks if c.id == chunk_id), "")),
            }
            for chunk_id, score, summary, reasoning in chunk_evaluations
        ]
        if chunk_evaluations
        else [],
    }

    # Verify the dataset structure before saving
    logger.info(f"Dataset structure for {model_to_use}/{query_key}:")
    logger.info(
        f"  Number of chunk IDs: {len(dataset['models'][model_to_use]['queries'][query_key]['chunk_ids'])}"
    )
    logger.info(
        f"  First 5 chunk IDs: {[id[:8] for id in dataset['models'][model_to_use]['queries'][query_key]['chunk_ids'][:5]]}"
    )

    # Log evaluation information if available
    if (
        "evaluations" in dataset["models"][model_to_use]["queries"][query_key]
        and dataset["models"][model_to_use]["queries"][query_key]["evaluations"]
    ):
        evaluations = dataset["models"][model_to_use]["queries"][query_key]["evaluations"]
        logger.info(f"  Number of evaluations: {len(evaluations)}")
        logger.info("  Top 3 evaluations:")
        for i, eval in enumerate(evaluations[:3]):
            logger.info(
                f"    #{i + 1}: ID={eval['chunk_id'][:8]}, Index={eval['index']}, Score={eval['score']}, Length={eval['chunk_len']}"
            )
            logger.info(f"      Body: {eval['body']}")
            logger.info(f"      Summary: {eval['summary'][:100]}...")
            logger.info(f"      Reasoning: {eval['reasoning'][:100]}...")

    # Save the updated dataset
    os.makedirs(dataset_path.parent, exist_ok=True)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    # Verify the file was written correctly
    if dataset_path.exists():
        file_size = os.path.getsize(dataset_path)
        logger.info(f"Saved golden dataset to {dataset_path} ({file_size} bytes)")

        # Read back the file to verify
        with open(dataset_path) as f:
            try:
                saved_dataset = json.load(f)
                saved_chunk_ids = saved_dataset["models"][model_to_use]["queries"][query_key][
                    "chunk_ids"
                ]
                saved_evaluations = saved_dataset["models"][model_to_use]["queries"][query_key].get(
                    "evaluations", []
                )
                logger.info(
                    f"Verified saved dataset: {len(saved_chunk_ids)} chunk IDs and {len(saved_evaluations)} evaluations for {model_to_use}/{query_key}"
                )
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error verifying saved dataset: {e}")
    else:
        logger.warning(f"Failed to save golden dataset to {dataset_path}")

    return chunk_ids


def calculate_tau_with_golden(golden_ids: list[str], test_ids: list[str]) -> float:
    """Calculate Kendall's Tau between a list of chunk IDs and the golden ranking.

    Args:
        golden_ids: The golden ranking of chunk IDs
        test_ids: The test ranking of chunk IDs

    Returns
    -------
        The Kendall's Tau correlation coefficient
    """
    # Find common elements
    common_ids = set(golden_ids).intersection(set(test_ids))

    if not common_ids:
        logger.warning("No common chunk IDs between golden and test rankings")
        return float("nan")

    # Log the number of common IDs
    logger.info(f"Number of common chunk IDs: {len(common_ids)} out of {len(golden_ids)}")

    # Get indices of common elements in both lists
    golden_indices = [golden_ids.index(chunk_id) for chunk_id in common_ids]
    test_indices = [test_ids.index(chunk_id) for chunk_id in common_ids]

    # Calculate Kendall's Tau
    tau, _ = kendalltau(golden_indices, test_indices)
    return tau


@pytest.fixture(
    params=[
        pytest.param(FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0), id="flashrank_english"),
        pytest.param(
            (
                ("en", FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0)),
                ("other", FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0)),
            ),
            id="flashrank_multilingual",
        ),
    ],
)
def reranker(
    request: pytest.FixtureRequest,
) -> BaseRanker | tuple[tuple[str, BaseRanker], ...]:
    """Get a reranker to test RAGLite with."""
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...] = request.param
    return reranker


@pytest.mark.parametrize(
    "golden_test_config",
    [
        {
            "documents": [
                "tests/paul_graham_essay.txt",
            ],  # "tests/agent.md", "tests/specrel.pdf", "tests/paul_graham_essay.txt",
            "test_cases": [
                # {
                #     "query": "What does it mean for two events to be simultaneous?",
                #     "document": "specrel.pdf",  # Default document
                #     "description": "Simultaneity in special relativity",
                # },
                {
                    "query": "What were the two main things the author worked on before college?",
                    "document": "paul_graham_essay.txt",
                    "description": "Paul Graham essay question",
                },
                # {
                #     "query": "What is Task Decomposition?",
                #     "document": "agent.md",
                #     "description": "Agent task decomposition question",
                # },
            ],
            "model": "gpt-4o",  # o3-mini, gpt-4o-mini
        }
    ],
    indirect=True,
)
def test_reranker_against_golden_parametrized(
    golden_test_config: tuple[RAGLiteConfig, list, str],
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...],
) -> None:
    """Test reranking against a golden dataset with multiple query-document pairs.

    Args:
        golden_test_config: Tuple containing (config, test_cases, model)
        reranker: The reranker to use (either English or Multilingual)
    """
    # Unpack the golden_test_config tuple
    config, test_cases, model = golden_test_config

    # Collect results from all test cases
    all_results = []

    # Run the test for each query-document pair
    for test_case in test_cases:
        result = _test_reranker_for_query(
            config=config,
            reranker=reranker,
            query=test_case["query"],
            document=test_case["document"],
            description=test_case["description"],
            model=model,
            assert_improvement=False,  # Don't assert in individual tests
        )
        all_results.append(result)

    # Calculate average tau values
    avg_search_tau = np.nanmean([r["search_tau"] for r in all_results])
    avg_reranked_tau = np.nanmean([r["reranked_tau"] for r in all_results])

    # Log the average results
    logger.info("=" * 50)
    logger.info("AVERAGE RESULTS ACROSS ALL TEST CASES")
    logger.info(f"Average τ_search_golden: {avg_search_tau:.4f}")
    logger.info(f"Average τ_reranked_golden: {avg_reranked_tau:.4f}")
    logger.info(f"Average improvement: {avg_reranked_tau - avg_search_tau:.4f}")
    logger.info("=" * 50)

    # Assert that reranking improves correlation with the golden ranking on average
    if np.isnan(avg_search_tau) or np.isnan(avg_reranked_tau):
        logger.warning(
            f"Skipping assertion due to NaN values in averages: "
            f"avg_search_tau={avg_search_tau}, avg_reranked_tau={avg_reranked_tau}"
        )
        skip("NaN values in average Kendall's Tau calculation")

    # If we get here, we have valid average tau values
    assert avg_reranked_tau > avg_search_tau, (
        f"Reranking should improve correlation with golden ranking on average: "
        f"avg_reranked_tau={avg_reranked_tau:.4f} <= avg_search_tau={avg_search_tau:.4f}"
    )


def _test_reranker_for_query(
    config: RAGLiteConfig,
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...],
    query: str,
    document: str | None = None,
    description: str = "",
    model: str = "gpt-4o-mini",
    assert_improvement: bool = True,
) -> dict:
    """Test reranking against a golden dataset for a specific query.

    Args:
        config: The RAGLite configuration to use
        reranker: The reranker to use (either English or Multilingual)
        query: The query to test
        document: Optional document filename to use for the test
        description: Description of the test case
        model: Model to use for ranking (defaults to gpt-4o-mini)
    """
    # We don't reset the golden dataset anymore to preserve rankings across test runs
    # This allows us to build up a comprehensive golden dataset over time

    # Get database type and embedder type for logging
    db_type = "postgres" if "postgres" in config.db_url else "sqlite"
    embedder = config.embedder

    # Get reranker type for logging
    reranker_type = (
        "flashrank_english" if isinstance(reranker, FlashRankRanker) else "flashrank_multilingual"
    )

    logger.info(f"Testing with reranker: {reranker_type}")
    logger.info(f"Test case: {description}")
    logger.info(f"Query: {query}")
    if document:
        logger.info(f"Document: {document}")

    # Update the config with the reranker
    config = RAGLiteConfig(
        db_url=config.db_url,
        embedder=config.embedder,
        reranker=reranker,
    )

    # Process the document if specified
    if document:
        document_path = Path(__file__).parent / document
        logger.info(f"Processing document: {document_path}")

        # Check if document exists
        if not document_path.exists():
            logger.error(f"Document not found: {document_path}")
            skip(f"Document not found: {document_path}")

        # Use the same approach as in conftest.py - insert the document and then search
        from raglite import insert_document

        # Insert the document
        insert_document(document_path, config=config)

        # Search for the query to get chunks
        chunk_ids, scores = hybrid_search(query, num_results=20, config=config)
        chunks = retrieve_chunks(chunk_ids, config=config)
    else:
        # If no document specified, use hybrid search to get chunks
        chunk_ids, scores = hybrid_search(query, num_results=20, config=config)
        chunks = retrieve_chunks(chunk_ids, config=config)

    # Get or create the golden ranking, always using the specified model (defaults to gpt-4o-mini)
    golden_chunk_ids = update_golden_dataset_with_real_chunks(
        query, chunks, config, model_override=model
    )

    # Rerank the chunks
    reranked_chunks = rerank_chunks(query, chunks, config=config)
    reranked_chunk_ids = [chunk.id for chunk in reranked_chunks]

    # Create a mapping of chunk IDs to indices and content for easier logging
    chunk_id_to_index = {chunk.id: chunk.index for chunk in chunks if hasattr(chunk, "index")}
    chunk_id_to_content = {chunk.id: chunk.body for chunk in chunks}

    # Create a log file for this test case
    log_dir = Path(__file__).parent / "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = (
        log_dir
        / f"ranking_comparison_{query.replace(' ', '_')[:30]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    # Log the chunk IDs and content for debugging
    with open(log_file, "w") as f:
        # Write test case information
        f.write(f"Query: {query}\n")
        f.write(f"Document: {document}\n")
        f.write(f"Description: {description}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Reranker: {reranker_type}\n\n")

        # Write golden ranking
        f.write("=" * 80 + "\n")
        f.write("GOLDEN RANKING (FIRST 5):\n")
        f.write("=" * 80 + "\n")
        for i, chunk_id in enumerate(golden_chunk_ids[:5]):
            index = chunk_id_to_index.get(chunk_id, "N/A")
            content = chunk_id_to_content.get(chunk_id, "Content not available")
            f.write(f"#{i}: Index={index}, ID={chunk_id[:8]}\n")
            f.write(f"Content:\n{content}\n\n")

        # Write search ranking
        f.write("=" * 80 + "\n")
        f.write("SEARCH RANKING (FIRST 5):\n")
        f.write("=" * 80 + "\n")
        for i, chunk_id in enumerate(chunk_ids[:5]):
            index = chunk_id_to_index.get(chunk_id, "N/A")
            content = chunk_id_to_content.get(chunk_id, "Content not available")
            f.write(f"#{i}: Index={index}, ID={chunk_id[:8]}\n")
            f.write(f"Content:\n{content}\n\n")

        # Write reranked ranking
        f.write("=" * 80 + "\n")
        f.write("RERANKED RANKING (FIRST 5):\n")
        f.write("=" * 80 + "\n")
        for i, chunk_id in enumerate(reranked_chunk_ids[:5]):
            index = chunk_id_to_index.get(chunk_id, "N/A")
            content = chunk_id_to_content.get(chunk_id, "Content not available")
            f.write(f"#{i}: Index={index}, ID={chunk_id[:8]}\n")
            f.write(f"Content:\n{content}\n\n")

    # Log to console that we've written the file
    logger.info(f"Wrote ranking comparison to {log_file}")

    # Also log the chunk IDs for debugging in the console
    logger.info("Golden ranking (first 5):")
    for i, chunk_id in enumerate(golden_chunk_ids[:5]):
        index = chunk_id_to_index.get(chunk_id, "N/A")
        logger.info(f"  #{i}: Index={index}, ID={chunk_id[:8]}")

    logger.info("Search ranking (first 5):")
    for i, chunk_id in enumerate(chunk_ids[:5]):
        index = chunk_id_to_index.get(chunk_id, "N/A")
        logger.info(f"  #{i}: Index={index}, ID={chunk_id[:8]}")

    logger.info("Reranked ranking (first 5):")
    for i, chunk_id in enumerate(reranked_chunk_ids[:5]):
        index = chunk_id_to_index.get(chunk_id, "N/A")
        logger.info(f"  #{i}: Index={index}, ID={chunk_id[:8]}")

    # Calculate Kendall's Tau against the golden ranking
    τ_search_golden = calculate_tau_with_golden(golden_chunk_ids, chunk_ids)
    τ_reranked_golden = calculate_tau_with_golden(golden_chunk_ids, reranked_chunk_ids)

    # Log the results
    logger.info(f"Test configuration: {db_type}-{embedder}")
    logger.info(f"τ_search_golden: {τ_search_golden:.4f}")
    logger.info(f"τ_reranked_golden: {τ_reranked_golden:.4f}")
    logger.info(f"Improvement: {τ_reranked_golden - τ_search_golden:.4f}")

    # Log the rankings
    logger.info("Golden ranking (first 5):")
    for i, chunk_id in enumerate(golden_chunk_ids[:5]):
        logger.info(f"  #{i}: {chunk_id[:8]}")

    logger.info("Search ranking (first 5):")
    for i, chunk_id in enumerate(chunk_ids[:5]):
        logger.info(f"  #{i}: {chunk_id[:8]}")

    logger.info("Reranked ranking (first 5):")
    for i, chunk_id in enumerate(reranked_chunk_ids[:5]):
        logger.info(f"  #{i}: {chunk_id[:8]}")

    # Assert that reranking improves correlation with the golden ranking
    # This is what we want to test: does reranking bring us closer to the ideal ranking?
    # Only assert if we have valid tau values and assert_improvement is True
    if assert_improvement:
        if np.isnan(τ_search_golden) or np.isnan(τ_reranked_golden):
            logger.warning(
                f"Skipping assertion due to NaN values: "
                f"τ_search_golden={τ_search_golden}, τ_reranked_golden={τ_reranked_golden}"
            )
            skip("NaN values in Kendall's Tau calculation")

        # If we get here, we have valid tau values
        assert τ_reranked_golden > τ_search_golden, (
            f"Reranking should improve correlation with golden ranking: "
            f"τ_reranked_golden={τ_reranked_golden:.4f} <= τ_search_golden={τ_search_golden:.4f}"
        )

    # Return the results for aggregation
    return {
        "search_tau": τ_search_golden,
        "reranked_tau": τ_reranked_golden,
        "improvement": τ_reranked_golden - τ_search_golden,
        "query": query,
        "document": document,
    }


def test_reranker_against_golden(
    raglite_test_config: RAGLiteConfig,
    reranker: BaseRanker | tuple[tuple[str, BaseRanker], ...],
    model: str = "gpt-4o-mini",
) -> None:
    """Legacy test function that tests only the original query.

    This is kept for backward compatibility.

    Args:
        raglite_test_config: The RAGLite configuration to use
        reranker: The reranker to use (either English or Multilingual)
        model: Model to use for ranking (defaults to gpt-4o-mini)
    """
    _test_reranker_for_query(
        config=raglite_test_config,
        reranker=reranker,
        query="What does it mean for two events to be simultaneous?",
        description="Original simultaneity test case",
        model=model,
    )


if __name__ == "__main__":
    # This allows running the test directly for debugging
    # Note: When running directly, we need to create a config manually
    # since we can't use pytest fixtures
    import sys

    from conftest import raglite_test_config as get_config

    # Parse command line arguments
    model = "gpt-4o-mini"  # Default to gpt-4o-mini
    reranker_type = "english"  # Default to English reranker
    test_case = 0  # Default to the first test case (simultaneity)
    run_all = False  # Flag to run all test cases and calculate average

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "all":
            run_all = True
            print("Running all test cases and calculating average")
        else:
            model = sys.argv[1]  # Allow specifying model as command line argument
            print(f"Using model override: {model}")

    if len(sys.argv) > 2 and not run_all:
        reranker_type = sys.argv[2]  # Allow specifying reranker type
        print(f"Using reranker type: {reranker_type}")

    if len(sys.argv) > 3 and not run_all:
        test_case = int(sys.argv[3])  # Allow specifying test case index
        print(f"Using test case index: {test_case}")

    # Create the appropriate reranker
    if reranker_type.lower() == "multilingual":
        reranker = (
            ("en", FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0)),
            ("other", FlashRankRanker("ms-marco-MultiBERT-L-12", verbose=0)),
        )
        print("Using multilingual reranker")
    else:
        reranker = FlashRankRanker("ms-marco-MiniLM-L-12-v2", verbose=0)
        print("Using English reranker")

    # Create a test config for SQLite with OpenAI embeddings
    db_url = "sqlite:///tmp/raglite_test_remote.sqlite"
    config = get_config(db_url, "gpt-4o-mini", "text-embedding-3-small")

    # Define the test cases
    test_cases = [
        {
            "query": "What does it mean for two events to be simultaneous?",
            "document": "specrel.pdf",
            "description": "Simultaneity in special relativity",
        },
        {
            "query": "What were the two main things the author worked on before college?",
            "document": "paul_graham_essay.txt",
            "description": "Paul Graham essay question",
        },
        {
            "query": "What is Task Decomposition?",
            "document": "agent.md",
            "description": "Agent task decomposition question",
        },
    ]

    if run_all:
        # Run all test cases and calculate average
        all_results = []
        for case in test_cases:
            print(f"Running test case: {case['description']}")
            result = _test_reranker_for_query(
                config,
                reranker,
                case["query"],
                case["document"],
                case["description"],
                model,
                assert_improvement=False,
            )
            all_results.append(result)

        # Calculate and print average results
        avg_search_tau = np.nanmean([r["search_tau"] for r in all_results])
        avg_reranked_tau = np.nanmean([r["reranked_tau"] for r in all_results])
        print("=" * 50)
        print("AVERAGE RESULTS ACROSS ALL TEST CASES")
        print(f"Average τ_search_golden: {avg_search_tau:.4f}")
        print(f"Average τ_reranked_golden: {avg_reranked_tau:.4f}")
        print(f"Average improvement: {avg_reranked_tau - avg_search_tau:.4f}")
        print("=" * 50)

        # Assert improvement on average
        if not (np.isnan(avg_search_tau) or np.isnan(avg_reranked_tau)):
            if avg_reranked_tau > avg_search_tau:
                print("✅ Reranking improves correlation with golden ranking on average")
            else:
                print("❌ Reranking does NOT improve correlation with golden ranking on average")
    # Run the selected test case
    elif test_case < len(test_cases):
        selected_case = test_cases[test_case]
        print(f"Running test case: {selected_case['description']}")
        _test_reranker_for_query(
            config,
            reranker,
            selected_case["query"],
            selected_case["document"],
            selected_case["description"],
            model,
        )
    else:
        print(f"Invalid test case index: {test_case}. Running all test cases individually.")
        for case in test_cases:
            print(f"Running test case: {case['description']}")
            _test_reranker_for_query(
                config, reranker, case["query"], case["document"], case["description"], model
            )

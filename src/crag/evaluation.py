# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import time
from datetime import datetime
from enum import Enum

from dotenv import load_dotenv
from loguru import logger
from openai import APIConnectionError, OpenAI, RateLimitError
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from crag.models import OpenAIRAGModel, RAGLiteModel
from crag.prompts.templates import IN_CONTEXT_EXAMPLES, INSTRUCTIONS

UserModel = RAGLiteModel | OpenAIRAGModel

load_dotenv()

TASK = "set"  # or "set, comparison, condition"
DATASET_PATH = os.getenv(f"EVALUATION_DATASET_PATH_{TASK.upper()}")
EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME")
OPENAI_API_KEY = os.getenv("EVALUATION_API_KEY")
OPENAI_BASE_URL = os.getenv("EVALUATION_API_BASE")

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


def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    logger.info(f"Loading JSON from {file_path}")
    with open(file_path) as f:
        return json.load(f)


def get_system_message():
    """Return the system message containing instructions and in context examples."""
    return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES


def attempt_api_call(client: OpenAI, model_name: str, messages: list, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # TODO: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=model_name,
                input=messages,
                text_format=CRAGResponse,
            )
            return response.output_parsed
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def log_response(messages, response, file_name):
    """Save the response from the API to a file."""
    with open(file_name, "a") as f:
        json.dump({"messages": messages[1], "response": response}, f)
        f.write("\n")


def load_data_in_batches(dataset_path, batch_size):
    """
    Read data from a compressed file and yields batches of data.

    Each batch is a dictionary containing lists of interaction_ids, queries, search results,
    query times, and answers.

    Args:
    dataset_path (str): Path to the dataset file.
    batch_size (int): Number of data items in each batch.

    Yields
    ------
    dict: A batch of data.
    """

    def initialize_batch():
        """Create an empty batch."""
        return {
            "interaction_id": [],
            "query": [],
            "search_results": [],
            "query_time": [],
            "answer": [],
        }

    try:
        with open(dataset_path) as file:
            batch = initialize_batch()
            for _, line in enumerate(file):
                try:
                    item = json.loads(line)
                    for key in batch:
                        batch[key].append(item[key])

                    if len(batch["query"]) == batch_size:
                        yield batch
                        batch = initialize_batch()
                except json.JSONDecodeError:  # noqa: PERF203
                    logger.warning("Warning: Failed to decode a line.")
            # Yield any remaining data as the last batch
            if batch["query"]:
                yield batch
    except FileNotFoundError as e:
        logger.error(f"Error: The file {dataset_path} was not found.")
        raise e
    except OSError as e:
        logger.error(f"Error: An error occurred while reading the file {dataset_path}.")
        raise e


def generate_predictions(dataset_path, participant_model: UserModel, output_file):
    """
    Process batches of data from a dataset to generate predictions using a model.

    Args:
    dataset_path (str): Path to the dataset.
    participant_model (UserModel): UserModel that provides `get_batch_size()` and
    `batch_generate_answer()` interfaces.

    Returns
    -------
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    length = 0
    with open(dataset_path) as file:
        for _ in file:
            length += 1

    for batch in tqdm(
        load_data_in_batches(dataset_path, batch_size),
        desc="Generating predictions...",
        total=(length + batch_size - 1) // batch_size,
    ):
        batch_ground_truths = batch.pop("answer")  # Remove answers from batch and store them
        batch_predictions, _ = participant_model.batch_generate_answer(batch)

        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)

        # append to output file incrementally
        for q, gt, pred in zip(
            batch["query"], batch_ground_truths, batch_predictions, strict=False
        ):
            with open(output_file, "a") as f:
                json_line = json.dumps({"query": q, "ground_truth": gt, "prediction": pred})
                f.write(json_line + "\n")

        time.sleep(1)  # avoid rate limiting

    return queries, ground_truths, predictions


def evaluate_predictions(
    queries, ground_truths, predictions, evaluation_model_name, save_file_name
):
    """
    Evaluate the predictions generated by a model against ground truth answers.

    Args:
    queries (List[str]): List of queries.
    ground_truths (List[str]): List of ground truth answers.
        Note each query can have multiple ground truth answers.
    predictions (list): List of predictions generated by the model.
    evaluation_model_name (str): Name of the evaluation model.

    Returns
    -------
    dict: A dictionary containing evaluation results.
    """
    # now we are using chatgpt
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    system_message = get_system_message()

    n_miss = 0
    n_correct = 0
    n_incorrect = 0

    # helper function to check if a prediction is considered "missing"
    def is_missing(pred: str) -> bool:
        p = (pred or "").strip()
        if not p:
            return True
        return bool(ABSTAIN_RE.search(p))

    for query, ground_truth, prediction in tqdm(
        zip(queries, ground_truths, predictions, strict=True),
        total=len(predictions),
        desc="Evaluating Predictions",
    ):
        ground_truth = ground_truth.strip()
        prediction = prediction.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n"
                f"Ground truth: {ground_truth}\n"
                f"Prediction: {prediction}\n",
            },
        ]

        # prefilter clear "missing" predictions to avoid unnecessary API calls
        if is_missing(prediction):
            n_miss += 1
            log_response(messages, '{"label": "missing"}', save_file_name)
            continue

        # prefilter exact matches to avoid unnecessary API calls
        if prediction_lowercase == ground_truth_lowercase:
            n_correct += 1
            log_response(messages, '{"label": "correct"}', save_file_name)
            continue

        # need to use the OpenAI evaluation model to get the accuracy result (0 means wrong, 1 means correct)
        response = attempt_api_call(openai_client, evaluation_model_name, messages)
        if response is None:
            n_miss += 1  # if all retries fail, consider it as missing (neutral)
            continue

        log_response(messages, response.model_dump_json(), save_file_name)
        label = response.label
        if label == CRAGLabel.missing:
            n_miss += 1
        elif label == CRAGLabel.correct:
            n_correct += 1
        elif label == CRAGLabel.incorrect:
            n_incorrect += 1
        else:
            # if all retries fail, consider it as missing (neutral)
            n_miss += 1

    # 3-way score in [-1, 1]: correct=+1, missing=0, incorrect=-1
    n = len(predictions)
    results = {
        "score": (n_correct - n_incorrect) / n,
        "accuracy": n_correct / n,
        "hallucination": n_incorrect / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_hallucination": n_incorrect,
        "total": n,
    }
    logger.info(results)

    # add results to log file
    with open(save_file_name, "a") as f:
        f.write(json.dumps(results) + "\n")
    return results


if __name__ == "__main__":
    #### Select model to evaluate
    model_id = "openai"
    participant_model = OpenAIRAGModel(
        TASK, vector_store_id=os.getenv(f"OPENAI_VECTOR_STORE_ID_{TASK.upper()}")
    )
    # participant_model.ingest_documents()

    # # Raglite
    # model_id = "raglite"
    # participant_model = RAGLiteModel(
    #     TASK,
    #     use_self_query=True,
    #     use_rerank=False,
    #     use_hybrid_search=False,
    #     # use_agentic_rag=False,
    # )
    # participant_model.ingest_documents()

    #### Generate predictions
    tz = datetime.now().astimezone().tzinfo
    output_file = f"{datetime.now(tz=tz).strftime('%Y%m%d-%H%M%S')}_{model_id}_preds.jsonl"
    queries, ground_truths, predictions = generate_predictions(
        DATASET_PATH, participant_model, output_file
    )

    #### Evaluate Predictions
    save_file_name = f"{datetime.now(tz=tz).strftime('%Y%m%d-%H%M%S')}_{model_id}_selfQ.jsonl"
    evaluation_results = evaluate_predictions(
        queries, ground_truths, predictions, EVALUATION_MODEL_NAME, save_file_name
    )

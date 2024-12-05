"""Generation and evaluation of evals."""

from random import randint
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlmodel import Session, func, select
from tqdm.auto import tqdm, trange

from raglite._config import RAGLiteConfig
from raglite._database import Chunk, Document, Eval, create_database_engine
from raglite._extract import extract_with_llm
from raglite._rag import create_rag_instruction, rag, retrieve_rag_context
from raglite._search import hybrid_search, retrieve_chunk_spans, vector_search
from raglite._typing import SearchMethod


def insert_evals(  # noqa: C901
    *, num_evals: int = 100, max_contexts_per_eval: int = 20, config: RAGLiteConfig | None = None
) -> None:
    """Generate and insert evals into the database."""

    class QuestionResponse(BaseModel):
        """A specific question about the content of a set of document contexts."""

        model_config = ConfigDict(
            extra="forbid"  # Forbid extra attributes as required by OpenAI's strict mode.
        )
        question: str = Field(
            ..., description="A specific question about the content of a set of document contexts."
        )
        system_prompt: ClassVar[str] = """
You are given a set of contexts extracted from a document.
You are a subject matter expert on the document's topic.
Your task is to generate a question to quiz other subject matter experts on the information in the provided context.
The question MUST satisfy ALL of the following criteria:
- The question SHOULD integrate as much of the provided context as possible.
- The question MUST NOT be a general or open question, but MUST instead be as specific to the provided context as possible.
- The question MUST be completely answerable using ONLY the information in the provided context, without depending on any background information.
- The question MUST be entirely self-contained and able to be understood in full WITHOUT access to the provided context.
- The question MUST NOT reference the existence of the context, directly or indirectly.
- The question MUST treat the context as if its contents are entirely part of your working memory.
            """.strip()

        @field_validator("question")
        @classmethod
        def validate_question(cls, value: str) -> str:
            """Validate the question."""
            question = value.strip().lower()
            if "context" in question or "document" in question or "question" in question:
                raise ValueError
            if not question.endswith("?"):
                raise ValueError
            return value

    config = config or RAGLiteConfig()
    engine = create_database_engine(config)
    with Session(engine) as session:
        for _ in trange(num_evals, desc="Generating evals", unit="eval", dynamic_ncols=True):
            # Sample a random document from the database.
            seed_document = session.exec(select(Document).order_by(func.random()).limit(1)).first()
            if seed_document is None:
                error_message = "First run `insert_document()` before generating evals."
                raise ValueError(error_message)
            # Sample a random chunk from that document.
            seed_chunk = session.exec(
                select(Chunk)
                .where(Chunk.document_id == seed_document.id)
                .order_by(func.random())
                .limit(1)
            ).first()
            if seed_chunk is None:
                continue
            # Expand the seed chunk into a set of related chunks.
            related_chunk_ids, _ = vector_search(
                query=np.mean(seed_chunk.embedding_matrix, axis=0, keepdims=True),
                num_results=randint(2, max_contexts_per_eval // 2),  # noqa: S311
                config=config,
            )
            related_chunks = [
                str(chunk_spans)
                for chunk_spans in retrieve_chunk_spans(related_chunk_ids, config=config)
            ]
            # Extract a question from the seed chunk's related chunks.
            try:
                question_response = extract_with_llm(
                    QuestionResponse, related_chunks, strict=True, config=config
                )
            except ValueError:
                continue
            else:
                question = question_response.question
            # Search for candidate chunks to answer the generated question.
            candidate_chunk_ids, _ = hybrid_search(
                query=question, num_results=max_contexts_per_eval, config=config
            )
            candidate_chunks = [session.get(Chunk, chunk_id) for chunk_id in candidate_chunk_ids]

            # Determine which candidate chunks are relevant to answer the generated question.
            class ContextEvalResponse(BaseModel):
                """Indicate whether the provided context can be used to answer a given question."""

                model_config = ConfigDict(
                    extra="forbid"  # Forbid extra attributes as required by OpenAI's strict mode.
                )
                hit: bool = Field(
                    ...,
                    description="True if the provided context contains (a part of) the answer to the given question, false otherwise.",
                )
                system_prompt: ClassVar[str] = f"""
You are given a context extracted from a document.
You are a subject matter expert on the document's topic.
Your task is to answer whether the provided context contains (a part of) the answer to this question: "{question}"
An example of a context that does NOT contain (a part of) the answer is a table of contents.
                    """.strip()

            relevant_chunks = []
            for candidate_chunk in tqdm(
                candidate_chunks, desc="Evaluating chunks", unit="chunk", dynamic_ncols=True
            ):
                try:
                    context_eval_response = extract_with_llm(
                        ContextEvalResponse, str(candidate_chunk), strict=True, config=config
                    )
                except ValueError:  # noqa: PERF203
                    pass
                else:
                    if context_eval_response.hit:
                        relevant_chunks.append(candidate_chunk)
            if not relevant_chunks:
                continue

            # Answer the question using the relevant chunks.
            class AnswerResponse(BaseModel):
                """Answer a question using the provided context."""

                model_config = ConfigDict(
                    extra="forbid"  # Forbid extra attributes as required by OpenAI's strict mode.
                )
                answer: str = Field(
                    ...,
                    description="A complete answer to the given question using the provided context.",
                )
                system_prompt: ClassVar[str] = f"""
You are given a set of contexts extracted from a document.
You are a subject matter expert on the document's topic.
Your task is to generate a complete answer to the following question using the provided context: "{question}"
The answer MUST satisfy ALL of the following criteria:
- The answer MUST integrate as much of the provided context as possible.
- The answer MUST be entirely self-contained and able to be understood in full WITHOUT access to the provided context.
- The answer MUST NOT reference the existence of the context, directly or indirectly.
- The answer MUST treat the context as if its contents are entirely part of your working memory.
                    """.strip()

            try:
                answer_response = extract_with_llm(
                    AnswerResponse,
                    [str(relevant_chunk) for relevant_chunk in relevant_chunks],
                    strict=True,
                    config=config,
                )
            except ValueError:
                continue
            else:
                answer = answer_response.answer
            # Store the eval in the database.
            eval_ = Eval.from_chunks(
                question=question, contexts=relevant_chunks, ground_truth=answer
            )
            session.add(eval_)
            session.commit()


def answer_evals(
    num_evals: int = 100,
    search: SearchMethod = hybrid_search,
    *,
    config: RAGLiteConfig | None = None,
) -> pd.DataFrame:
    """Read evals from the database and answer them with RAG."""
    # Read evals from the database.
    config = config or RAGLiteConfig()
    engine = create_database_engine(config)
    with Session(engine) as session:
        evals = session.exec(select(Eval).limit(num_evals)).all()
    # Answer evals with RAG.
    answers: list[str] = []
    contexts: list[list[str]] = []
    for eval_ in tqdm(evals, desc="Answering evals", unit="eval", dynamic_ncols=True):
        chunk_spans = retrieve_rag_context(query=eval_.question, search=search, config=config)
        messages = [create_rag_instruction(user_prompt=eval_.question, context=chunk_spans)]
        response = rag(messages, config=config)
        answer = "".join(response)
        answers.append(answer)
        chunk_ids, _ = search(query=eval_.question, config=config)
        contexts.append(
            [str(chunk_span) for chunk_span in retrieve_chunk_spans(chunk_ids, config=config)]
        )
    # Collect the answered evals.
    answered_evals: dict[str, list[str] | list[list[str]]] = {
        "question": [eval_.question for eval_ in evals],
        "answer": answers,
        "contexts": contexts,
        "ground_truth": [eval_.ground_truth for eval_ in evals],
        "ground_truth_contexts": [eval_.contexts for eval_ in evals],
    }
    answered_evals_df = pd.DataFrame.from_dict(answered_evals)
    return answered_evals_df


def evaluate(
    answered_evals: pd.DataFrame | int = 100, config: RAGLiteConfig | None = None
) -> pd.DataFrame:
    """Evaluate the performance of a set of answered evals with Ragas."""
    try:
        from datasets import Dataset
        from langchain_community.chat_models import ChatLiteLLM
        from langchain_community.llms import LlamaCpp
        from ragas import RunConfig
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings import BaseRagasEmbeddings

        from raglite._config import RAGLiteConfig
        from raglite._embed import embed_sentences
        from raglite._litellm import LlamaCppPythonLLM
    except ImportError as import_error:
        error_message = "To use the `evaluate` function, please install the `ragas` extra."
        raise ImportError(error_message) from import_error

    class RAGLiteRagasEmbeddings(BaseRagasEmbeddings):
        """A RAGLite embedder for Ragas."""

        def __init__(self, config: RAGLiteConfig | None = None):
            self.config = config or RAGLiteConfig()

        def embed_query(self, text: str) -> list[float]:
            # Embed the input text with RAGLite's embedding function.
            embeddings = embed_sentences([text], config=self.config)
            return embeddings[0].tolist()  # type: ignore[no-any-return]

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # Embed a list of documents with RAGLite's embedding function.
            embeddings = embed_sentences(texts, config=self.config)
            return embeddings.tolist()  # type: ignore[no-any-return]

    # Create a set of answered evals if not provided.
    config = config or RAGLiteConfig()
    answered_evals_df = (
        answered_evals
        if isinstance(answered_evals, pd.DataFrame)
        else answer_evals(num_evals=answered_evals, config=config)
    )
    # Load the LLM.
    if config.llm.startswith("llama-cpp-python"):
        llm = LlamaCppPythonLLM().llm(model=config.llm)
        lc_llm = LlamaCpp(
            model_path=llm.model_path,
            n_batch=llm.n_batch,
            n_ctx=llm.n_ctx(),
            n_gpu_layers=-1,
            verbose=llm.verbose,
        )
    else:
        lc_llm = ChatLiteLLM(model=config.llm)  # type: ignore[call-arg]
    embedder = RAGLiteRagasEmbeddings(config=config)
    # Evaluate the answered evals with Ragas.
    evaluation_df = ragas_evaluate(
        dataset=Dataset.from_pandas(answered_evals_df),
        llm=lc_llm,
        embeddings=embedder,
        run_config=RunConfig(max_workers=1),
    ).to_pandas()
    return evaluation_df

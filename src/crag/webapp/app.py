from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import streamlit as st

from crag.prompts.templates import (
    COMPARISON_IN_CONTEXT_EXAMPLES,
    COMPARISON_INSTRUCTIONS,
    SET_IN_CONTEXT_EXAMPLES,
    SET_INSTRUCTIONS,
)
from crag.webapp.services.judge_service import CRAGLabel, run_judge_once
from crag.webapp.services.openai_service import run_openai
from crag.webapp.services.raglite_service import run_raglite
from crag.webapp.ui.components import render_openai, render_raglite, render_search_results
from crag.webapp.utils.common import (
    _get_env,
    build_ground_truth_filename_keys,
    build_judge_system_message,
    format_metadata,
    is_missing_prediction,
)
from crag.webapp.utils.dataset import load_dataset_file


@st.cache_data(show_spinner=False)
def load_dataset_cached(path: str) -> list[dict[str, Any]]:
    return load_dataset_file(path)


def _get_judge_prompt_defaults(task: str) -> tuple[str, str]:
    if task == "set":
        return SET_INSTRUCTIONS, SET_IN_CONTEXT_EXAMPLES
    return COMPARISON_INSTRUCTIONS, COMPARISON_IN_CONTEXT_EXAMPLES


def _clear_prediction_and_judge_state() -> None:
    for key in [
        "raglite_result",
        "openai_result",
        "prediction_source",
        "run_judge",
    ]:
        st.session_state.pop(key, None)


def _render_notebook_comparison_model(result: dict[str, Any]) -> None:
    st.markdown(
        f"**Config:** `self_query={result.get('self_query')}` "
        f"`rerank={result.get('rerank')}` "
        f"`hybrid_search={result.get('hybrid_search')}`"
    )
    st.markdown("**Model answer:**")
    st.write(result.get("answer", ""))
    stdout = (result.get("stdout") or "").strip()
    stderr = (result.get("stderr") or "").strip()
    if stdout:
        st.markdown("**stdout:**")
        st.code(stdout)
    if stderr:
        st.markdown("**stderr:**")
        st.code(stderr)
    st.markdown("**Retrieved chunks:**")
    chunks = result.get("chunks", [])
    if not chunks:
        st.info("No chunks retrieved.")
    for idx, chunk in enumerate(chunks, start=1):
        filename = str(chunk.get("filename", "") or "").strip()
        url = str(chunk.get("url", "") or "").strip()
        metadata = chunk.get("metadata")
        domain = ""
        if isinstance(metadata, dict):
            domain = str(metadata.get("domain", "") or "").strip()
        title = filename or url or f"Chunk {idx}"
        domain_suffix = f" (domain: {domain})" if domain else ""
        with st.expander(f"{idx}. {title}{domain_suffix}", expanded=False):
            if isinstance(metadata, dict) and metadata:
                st.markdown("**Document metadata:**")
                st.code(format_metadata(metadata))
            st.markdown("**Chunk content:**")
            st.write(chunk.get("content", ""))


def _render_notebook_comparison_page() -> None:
    st.caption("Compare two independently configured RAGLite instances on the same sample.")

    with st.sidebar:
        st.header("Dataset")
        dataset_path = st.text_input(
            "Dataset Path",
            value=_get_env("EVALUATION_DATASET_PATH_COMPARISON", "data/crag_comparison_dev.jsonl"),
            key="comparison_page_dataset_path",
        )
        task = st.selectbox(
            "Task",
            ["comparison", "set"],
            index=0,
            key="comparison_page_task",
        )

    try:
        dataset = load_dataset_cached(dataset_path)
    except Exception as exc:
        st.error(str(exc))
        return

    if not dataset:
        st.warning("Dataset is empty.")
        return

    index = st.number_input(
        "Index",
        min_value=0,
        max_value=len(dataset) - 1,
        value=0,
        step=1,
        key="comparison_page_index",
    )
    sample = dataset[int(index)]
    sample_cache_key = (
        sample.get("interaction_id")
        or f"{dataset_path}|{task}|{int(index)}|{sample.get('query', '')}"
    )
    previous_sample_cache_key = st.session_state.get("_comparison_sample_cache_key")
    if previous_sample_cache_key != sample_cache_key:
        st.session_state.pop("notebook_comparison_result", None)
        st.session_state["_comparison_sample_cache_key"] = sample_cache_key

    st.subheader("Query")
    st.write(sample.get("query", ""))
    st.markdown(
        f"**Interaction ID:** {sample.get('interaction_id', '')}  \n"
        f"**Domain:** {sample.get('domain', '')}  \n"
        f"**Question Type:** {sample.get('question_type', '')}  \n"
        f"**Query Time:** {sample.get('query_time', '')}"
    )
    if sample.get("answer"):
        st.markdown("**Ground Truth Answer:**")
        st.write(sample["answer"])

    render_search_results(sample.get("search_results", []))

    st.divider()
    st.subheader("Comparison Setup")
    with st.expander("Show/Hide Setup", expanded=False):
        setup_col_a, setup_col_b = st.columns(2)
        with setup_col_a:
            st.markdown("**RAGLite A**")
            num_chunks_a = st.slider(
                "Chunks per Retrieval (A)",
                min_value=1,
                max_value=20,
                value=5,
                key="comparison_page_num_chunks_a",
            )
            self_query_a = st.checkbox(
                "Self Query (A)", value=True, key="comparison_page_self_query_a"
            )
            agentic_a = st.checkbox(
                "Agentic behavior (A)", value=True, key="comparison_page_agentic_a"
            )
            rerank_a = st.checkbox("Rerank (A)", value=False, key="comparison_page_rerank_a")
            hybrid_search_a = st.checkbox(
                "Hybrid Search (A)", value=False, key="comparison_page_hybrid_search_a"
            )
            raglite_db_url_a = st.text_input(
                "RAGLite DB URL (A)",
                value=_get_env(
                    f"RAGLITE_DB_URL_{task.upper()}",
                    _get_env("DB_RAGLITE_DB_URL_COMPARISONURL", ""),
                ),
                key="comparison_page_raglite_db_url_a",
            )
            raglite_embedder_a = st.text_input(
                "RAGLite Embedder (A)",
                value=_get_env("RAGLITE_EMBEDDER", ""),
                key="comparison_page_raglite_embedder_a",
            )
            raglite_llm_a = st.text_input(
                "RAGLite LLM (A)",
                value=_get_env("RAGLITE_LLM", ""),
                key="comparison_page_raglite_llm_a",
            )

        with setup_col_b:
            st.markdown("**RAGLite B**")
            num_chunks_b = st.slider(
                "Chunks per Retrieval (B)",
                min_value=1,
                max_value=20,
                value=5,
                key="comparison_page_num_chunks_b",
            )
            self_query_b = st.checkbox(
                "Self Query (B)", value=False, key="comparison_page_self_query_b"
            )
            agentic_b = st.checkbox(
                "Agentic behavior (B)", value=False, key="comparison_page_agentic_b"
            )
            rerank_b = st.checkbox("Rerank (B)", value=False, key="comparison_page_rerank_b")
            hybrid_search_b = st.checkbox(
                "Hybrid Search (B)", value=False, key="comparison_page_hybrid_search_b"
            )
            raglite_db_url_b = st.text_input(
                "RAGLite DB URL (B)",
                value=_get_env(
                    f"RAGLITE_DB_URL_{task.upper()}",
                    _get_env("DB_RAGLITE_DB_URL_COMPARISONURL", ""),
                ),
                key="comparison_page_raglite_db_url_b",
            )
            raglite_embedder_b = st.text_input(
                "RAGLite Embedder (B)",
                value=_get_env("RAGLITE_EMBEDDER", ""),
                key="comparison_page_raglite_embedder_b",
            )
            raglite_llm_b = st.text_input(
                "RAGLite LLM (B)",
                value=_get_env("RAGLITE_LLM", ""),
                key="comparison_page_raglite_llm_b",
            )

    if st.button("Run Comparison", key="comparison_page_run"):
        with st.spinner("Running RAGLite A/B..."):
            try:
                result_a = run_raglite(
                    query=sample.get("query", ""),
                    query_time=sample.get("query_time", ""),
                    task=task,
                    num_chunks=num_chunks_a,
                    db_url=raglite_db_url_a,
                    embedder=raglite_embedder_a,
                    llm=raglite_llm_a,
                    self_query=self_query_a,
                    rerank=rerank_a,
                    hybrid_search=hybrid_search_a,
                    capture_logs=True,
                    serialize_chunks=True,
                    agentic_rag=agentic_a,
                )

                result_b = run_raglite(
                    query=sample.get("query", ""),
                    query_time=sample.get("query_time", ""),
                    task=task,
                    num_chunks=num_chunks_b,
                    db_url=raglite_db_url_b,
                    embedder=raglite_embedder_b,
                    llm=raglite_llm_b,
                    self_query=self_query_b,
                    rerank=rerank_b,
                    hybrid_search=hybrid_search_b,
                    capture_logs=True,
                    serialize_chunks=True,
                    agentic_rag=agentic_b,
                )

                st.session_state["notebook_comparison_result"] = {
                    "instance_a": result_a,
                    "instance_b": result_b,
                }
            except Exception as exc:
                st.error(f"RAGLite comparison failed: {exc}")

    comparison_result = st.session_state.get("notebook_comparison_result")
    if not comparison_result:
        st.info("Run comparison to view both model outputs.")
        return
    if "instance_a" not in comparison_result or "instance_b" not in comparison_result:
        st.session_state.pop("notebook_comparison_result", None)
        st.info("Run comparison to view both model outputs.")
        return

    st.divider()
    st.subheader("Comparison Results")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("RAGLite A")
        _render_notebook_comparison_model(comparison_result["instance_a"])
    with col_b:
        st.subheader("RAGLite B")
        _render_notebook_comparison_model(comparison_result["instance_b"])


def main() -> None:
    st.set_page_config(page_title="CRAG Debugger", layout="wide")

    with st.sidebar:
        page = st.radio(
            "Page",
            options=["Judge Debugger", "RAGLite Comparison"],
            index=0,
            key="active_page",
        )

    st.title(page)

    if page == "RAGLite Comparison":
        _render_notebook_comparison_page()
        return

    with st.sidebar:
        st.header("Dataset")
        dataset_path = st.text_input(
            "Dataset Path",
            value=_get_env("EVALUATION_DATASET_PATH_COMPARISON", "data/crag_comparison_dev.jsonl"),
        )
        task = st.selectbox("Task", ["comparison", "set"], index=0)
        num_chunks = st.slider("Chunks per Retrieval", min_value=1, max_value=20, value=5)

        st.header("RAGLite")
        raglite_db_url = st.text_input(
            "RAGLite DB URL",
            value=_get_env(f"RAGLITE_DB_URL_{task.upper()}", ""),
        )
        raglite_embedder = st.text_input(
            "RAGLite Embedder",
            value=_get_env("RAGLITE_EMBEDDER", ""),
        )
        raglite_llm = st.text_input("RAGLite LLM", value=_get_env("RAGLITE_LLM", ""))
        raglite_hybrid_search = st.checkbox("Hybrid Search", value=False)

        st.header("OpenAI")
        openai_vector_store_id = st.text_input(
            "Vector Store ID",
            value=_get_env("OPENAI_VECTOR_STORE_ID", ""),
        )
        openai_model = st.text_input(
            "Model",
            value=_get_env("AZURE_LLM_DEPLOYMENT", "gpt-5-mini"),
        )
        openai_base_url = st.text_input(
            "Base URL",
            value=_get_env("AZURE_API_BASE", ""),
            help="Leave blank to use the OpenAI default base URL.",
        )
        openai_api_key = st.text_input(
            "API Key",
            value=_get_env("AZURE_API_KEY", ""),
            type="password",
        )

        st.header("Judge")
        judge_model = st.text_input(
            "Judge Model",
            value=_get_env("EVALUATION_MODEL_NAME", "gpt-5-mini"),
        )
        judge_base_url = st.text_input(
            "Judge Base URL",
            value=_get_env("EVALUATION_API_BASE", ""),
            help="Leave blank to use the OpenAI default base URL.",
        )
        judge_api_key = st.text_input(
            "Judge API Key",
            value=_get_env("EVALUATION_API_KEY", ""),
            type="password",
        )

    try:
        dataset = load_dataset_cached(dataset_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not dataset:
        st.warning("Dataset is empty.")
        st.stop()

    index = st.number_input("Index", min_value=0, max_value=len(dataset) - 1, value=0, step=1)
    sample = dataset[int(index)]
    sample_cache_key = (
        sample.get("interaction_id")
        or f"{dataset_path}|{task}|{int(index)}|{sample.get('query', '')}"
    )
    previous_sample_cache_key = st.session_state.get("_active_sample_cache_key")
    if previous_sample_cache_key != sample_cache_key:
        _clear_prediction_and_judge_state()
        st.session_state["_active_sample_cache_key"] = sample_cache_key

    st.subheader("Query")
    st.write(sample.get("query", ""))
    st.markdown(
        f"**Interaction ID:** {sample.get('interaction_id', '')}  \n"
        f"**Domain:** {sample.get('domain', '')}  \n"
        f"**Question Type:** {sample.get('question_type', '')}  \n"
        f"**Query Time:** {sample.get('query_time', '')}"
    )

    if sample.get("answer"):
        st.markdown("**Ground Truth Answer:**")
        st.write(sample["answer"])

    sample_search_results = sample.get("search_results", [])
    ground_truth_filename_keys = build_ground_truth_filename_keys(sample_search_results)

    render_search_results(sample_search_results)

    col_left, col_right = st.columns(2)

    with col_left:
        run_raglite_btn = st.button("Run RAGLite")

    with col_right:
        run_openai_btn = st.button("Run OpenAI")

    if st.button("Run Both"):
        run_raglite_btn = True
        run_openai_btn = True

    if run_raglite_btn:
        with st.spinner("Running RAGLite..."):
            try:
                raglite_result = run_raglite(
                    sample.get("query", ""),
                    query_time=sample.get("query_time", ""),
                    task=task,
                    num_chunks=num_chunks,
                    db_url=raglite_db_url,
                    embedder=raglite_embedder,
                    llm=raglite_llm,
                    hybrid_search=raglite_hybrid_search,
                    agentic_rag=True,
                )
                st.session_state["raglite_result"] = raglite_result
            except Exception as exc:
                st.error(f"RAGLite failed: {exc}")

    if run_openai_btn:
        if not openai_vector_store_id:
            st.error("OpenAI Vector Store ID is required.")
        elif not openai_api_key:
            st.error("OpenAI API Key is required.")
        else:
            with st.spinner("Running OpenAI..."):
                try:
                    openai_result = run_openai(
                        sample.get("query", ""),
                        query_time=sample.get("query_time", ""),
                        model_name=openai_model,
                        vector_store_id=openai_vector_store_id,
                        num_chunks=num_chunks,
                        base_url=openai_base_url,
                        api_key=openai_api_key,
                    )
                    st.session_state["openai_result"] = openai_result
                except Exception as exc:
                    st.error(f"OpenAI failed: {exc}")

    results_col_left, results_col_right = st.columns(2)

    with results_col_left:
        if "raglite_result" in st.session_state:
            render_raglite(
                st.session_state["raglite_result"]["chunks"],
                st.session_state["raglite_result"]["answer"],
                ground_truth_filename_keys=ground_truth_filename_keys,
            )

    with results_col_right:
        if "openai_result" in st.session_state:
            render_openai(
                st.session_state["openai_result"]["results"],
                st.session_state["openai_result"]["answer"],
                st.session_state["openai_result"]["citations"],
                st.session_state["openai_result"]["tool_results"],
                ground_truth_filename_keys=ground_truth_filename_keys,
            )

    st.divider()
    st.subheader("Judge Evaluation")

    default_judge_instructions, default_judge_examples = _get_judge_prompt_defaults(task)
    judge_instructions_key = f"judge_instructions_{task}"
    judge_examples_key = f"judge_examples_{task}"
    if judge_instructions_key not in st.session_state:
        st.session_state[judge_instructions_key] = default_judge_instructions
    if judge_examples_key not in st.session_state:
        st.session_state[judge_examples_key] = default_judge_examples

    with st.expander("Judge Prompt", expanded=False):
        judge_instructions = st.text_area(
            "Judge Instructions",
            key=judge_instructions_key,
            height=200,
        )
        judge_examples = st.text_area(
            "Judge In-Context Examples",
            key=judge_examples_key,
            height=300,
        )

    system_message = build_judge_system_message(judge_instructions, judge_examples)
    judge_runs = st.number_input("Judge Runs", min_value=1, max_value=20, value=3, step=1)
    use_short_circuit = st.checkbox(
        "Short-circuit missing/exact matches",
        value=True,
        help="If enabled, obvious missing or exact-match predictions skip the judge call.",
    )

    prediction_sources: list[str] = []
    if "raglite_result" in st.session_state:
        prediction_sources.append("RAGLite")
    if "openai_result" in st.session_state:
        prediction_sources.append("OpenAI")

    prediction_source = None
    if prediction_sources:
        prediction_source = st.selectbox("Prediction Source", prediction_sources, index=0)
    else:
        st.info("Run RAGLite or OpenAI to generate a prediction for evaluation.")

    run_judge_btn = st.button("Run Judge", disabled=prediction_source is None)

    if run_judge_btn:
        if not sample.get("answer"):
            st.error("Ground truth answer is missing for this sample.")
        elif not judge_api_key:
            st.error("Judge API Key is required.")
        else:
            if prediction_source == "RAGLite":
                prediction = st.session_state["raglite_result"]["answer"]
            else:
                prediction = st.session_state["openai_result"]["answer"]

            query = sample.get("query", "")
            ground_truth = sample.get("answer", "")

            results: list[dict[str, Any]] = []
            errors: list[str] = []

            if use_short_circuit and is_missing_prediction(prediction):
                for run_idx in range(int(judge_runs)):
                    results.append(
                        {
                            "run": run_idx + 1,
                            "label": CRAGLabel.missing.value,
                            "explanation": "Prediction is empty or abstains.",
                        }
                    )
            elif (
                use_short_circuit
                and (prediction or "").strip().lower() == (ground_truth or "").strip().lower()
            ):
                for run_idx in range(int(judge_runs)):
                    results.append(
                        {
                            "run": run_idx + 1,
                            "label": CRAGLabel.correct.value,
                            "explanation": "Exact match to ground truth.",
                        }
                    )
            else:
                total_runs = int(judge_runs)
                progress = st.progress(0, text="Starting judge runs...")
                with ThreadPoolExecutor(max_workers=min(8, total_runs)) as executor:
                    futures = {}
                    for run_idx in range(total_runs):
                        future = executor.submit(
                            run_judge_once,
                            query=query,
                            ground_truth=ground_truth,
                            prediction=prediction,
                            system_message=system_message,
                            model_name=judge_model,
                            base_url=judge_base_url,
                            api_key=judge_api_key,
                        )
                        futures[future] = run_idx + 1

                    completed = 0
                    for future in as_completed(futures):
                        run_id = futures[future]
                        try:
                            judge_result = future.result()
                            results.append(
                                {
                                    "run": run_id,
                                    "label": judge_result["label"],
                                    "explanation": judge_result["explanation"],
                                }
                            )
                        except Exception as exc:
                            errors.append(f"Run {run_id}: {exc}")
                        completed += 1
                        progress.progress(
                            int((completed / total_runs) * 100),
                            text=f"Completed {completed}/{total_runs}",
                        )
                progress.progress(100, text="Judge runs complete.")

            if results:
                st.markdown("**Judge Results:**")
                st.dataframe(results, use_container_width=True)

                counts = {label.value: 0 for label in CRAGLabel}
                for row in results:
                    label = row["label"]
                    counts[label] = counts.get(label, 0) + 1

                total = sum(counts.values())
                st.markdown(
                    f"**Summary:** correct={counts['correct']} "
                    f"incorrect={counts['incorrect']} missing={counts['missing']} "
                    f"(total={total})"
                )

            if errors:
                st.error("Some runs failed:")
                for err in errors:
                    st.write(err)


if __name__ == "__main__":
    main()

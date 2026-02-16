from typing import Any

from crag.webapp.utils.common import normalize_base_url


def run_openai(
    query: str,
    *,
    query_time: str,
    model_name: str,
    vector_store_id: str,
    num_chunks: int,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Run: pip install -r requirements.txt"
        ) from exc

    client = OpenAI(base_url=normalize_base_url(base_url) or None, api_key=api_key)
    search_results = None
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information "
                "from a collection of documents. Provide concise, accurate, and "
                "helpful answers based on the context provided by the file search results.\n\n"
                "Use this information as the primary source when forming your response.\n"
                f"Today is {query_time}",
            },
            {"role": "user", "content": query},
        ],
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": num_chunks,
            }
        ],
        max_tool_calls=1,
        parallel_tool_calls=False,
        tool_choice="required",
        include=["file_search_call.results"],
    )
    tool_results = extract_file_search_results_from_response(response)
    return {
        "results": search_results,
        "answer": response.output_text,
        "citations": extract_citations_from_response(response),
        "tool_results": tool_results,
    }


def extract_citations_from_response(response: Any) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            for annotation in getattr(content, "annotations", []) or []:
                ann_type = getattr(annotation, "type", None)
                if ann_type == "file_citation":
                    citations.append(
                        {
                            "type": ann_type,
                            "file_id": getattr(annotation, "file_id", ""),
                            "filename": getattr(annotation, "filename", ""),
                        }
                    )
                elif ann_type == "container_file_citation":
                    citations.append(
                        {
                            "type": ann_type,
                            "file_id": getattr(annotation, "file_id", ""),
                            "filename": getattr(annotation, "filename", ""),
                            "container_id": getattr(annotation, "container_id", ""),
                        }
                    )
                elif ann_type == "file_path":
                    citations.append(
                        {
                            "type": ann_type,
                            "file_id": getattr(annotation, "file_id", ""),
                            "index": getattr(annotation, "index", ""),
                        }
                    )
                elif ann_type == "url_citation":
                    citations.append(
                        {
                            "type": ann_type,
                            "title": getattr(annotation, "title", ""),
                            "url": getattr(annotation, "url", ""),
                        }
                    )
    return citations


def extract_file_search_results_from_response(response: Any) -> list[Any]:
    results: list[Any] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "file_search_call":
            continue
        for result in getattr(item, "results", []) or []:
            results.append(result)
    return results

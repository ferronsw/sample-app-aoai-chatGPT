import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Optional

from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.aio import SearchClient
from openai import AsyncAzureOpenAI


DEFAULT_NO_RESULTS_MESSAGE = (
    "I couldn't find relevant information in the indexed documents to answer that."
)


@dataclass
class SearchGroundingResult:
    citations: list[dict[str, Any]]
    intent: str
    prompt: Optional[str]
    query_type: str
    assistant_fallback: Optional[str] = None

    def tool_message(self) -> dict[str, Any]:
        return {
            "citations": self.citations,
            "intent": self.intent,
        }


def extract_user_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_segments = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text", "")
                if text_value:
                    text_segments.append(text_value.strip())
        return "\n".join(segment for segment in text_segments if segment)

    return ""


def get_latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            query_text = extract_user_text(message.get("content"))
            if query_text:
                return query_text

    return ""


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_citations = set()
    deduped_citations = []

    for citation in citations:
        citation_key = (
            citation.get("filepath") or "",
            citation.get("title") or "",
            citation.get("content") or "",
        )
        if citation_key in seen_citations:
            continue

        seen_citations.add(citation_key)
        deduped_citations.append(citation)

    return deduped_citations


def _get_caption_text(search_document: dict[str, Any]) -> Optional[str]:
    captions = search_document.get("@search.captions") or []
    caption_parts = []

    for caption in captions:
        text_value = None
        if isinstance(caption, dict):
            text_value = caption.get("text")
        else:
            text_value = getattr(caption, "text", None)

        if text_value:
            caption_parts.append(text_value.strip())

    if caption_parts:
        return " ... ".join(caption_parts)

    return None


def _get_content_text(search_document: dict[str, Any], search_settings) -> str:
    if search_settings.use_semantic_captions:
        caption_text = _get_caption_text(search_document)
        if caption_text:
            return caption_text

    content_parts = []
    for field_name in search_settings.content_columns or []:
        field_value = search_document.get(field_name)
        if isinstance(field_value, list):
            field_value = "\n".join(str(item) for item in field_value if item)
        if field_value:
            content_parts.append(str(field_value).strip())

    return "\n\n".join(part for part in content_parts if part)


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 3].rstrip() + "..."


def _format_metadata(search_document: dict[str, Any]) -> Optional[str]:
    metadata_parts = []
    score = search_document.get("@search.score")
    reranker_score = search_document.get("@search.reranker_score")

    if isinstance(score, (int, float)):
        metadata_parts.append(f"score={score:.4f}")
    if isinstance(reranker_score, (int, float)):
        metadata_parts.append(f"reranker_score={reranker_score:.4f}")

    if metadata_parts:
        return ", ".join(metadata_parts)

    return None


def _build_citation(search_document: dict[str, Any], search_settings, citation_index: int) -> Optional[dict[str, Any]]:
    content = _get_content_text(search_document, search_settings)
    if not content:
        return None

    filepath = None
    if search_settings.filename_column:
        filepath_value = search_document.get(search_settings.filename_column)
        filepath = str(filepath_value) if filepath_value else None

    title = None
    if search_settings.title_column:
        title_value = search_document.get(search_settings.title_column)
        title = str(title_value) if title_value else None

    url = None
    if search_settings.url_column:
        url_value = search_document.get(search_settings.url_column)
        url = str(url_value) if url_value else None

    citation = {
        "content": _truncate_text(content, search_settings.grounding_max_document_chars),
        "id": str(citation_index),
        "title": title,
        "filepath": filepath,
        "url": url,
        "metadata": _format_metadata(search_document),
        "chunk_id": str(search_document.get("chunk_id")) if search_document.get("chunk_id") is not None else None,
    }
    return citation


def _build_grounding_prompt(citations: list[dict[str, Any]], search_settings) -> tuple[Optional[str], list[dict[str, Any]]]:
    prompt_sections = []
    prompt_citations = []
    consumed_tokens = 0

    for index, citation in enumerate(citations, start=1):
        content = (citation.get("content") or "").strip()
        if not content:
            continue

        approx_tokens = max(1, len(content) // 4)
        if prompt_sections and consumed_tokens + approx_tokens > search_settings.grounding_max_tokens:
            break

        consumed_tokens += approx_tokens
        prompt_citations.append(citation)

        source_name = citation.get("title") or citation.get(
            "filepath") or f"Document {index}"
        source_details = [source_name]
        if citation.get("filepath") and citation.get("filepath") != source_name:
            source_details.append(f"path: {citation['filepath']}")
        if citation.get("url"):
            source_details.append(f"url: {citation['url']}")

        prompt_sections.append(
            f"[doc{index}] {' | '.join(source_details)}\n{content}"
        )

    if not prompt_sections:
        return None, []

    instruction_lines = [
        "Use the retrieved sources below to answer the latest user question.",
        "Cite grounded claims inline with [docN] markers that match the source list.",
        "Do not cite a source you did not use.",
    ]
    if search_settings.enable_in_domain:
        instruction_lines.append(
            "If the sources do not contain the answer, say that the information is not available in the indexed documents."
        )
    else:
        instruction_lines.append(
            "If you add general knowledge beyond the sources, clearly label it as ungrounded."
        )

    prompt = "\n".join(instruction_lines)
    prompt += "\n\nRetrieved sources:\n\n"
    prompt += "\n\n".join(prompt_sections)
    return prompt, prompt_citations


async def _create_embedding(query_text: str, app_settings) -> list[float]:
    embedding_model = app_settings.azure_openai.embedding_name
    if not embedding_model:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_NAME is required when AZURE_SEARCH_QUERY_TYPE uses vector retrieval."
        )

    api_key = app_settings.azure_openai.embedding_key or app_settings.azure_openai.key
    endpoint = app_settings.azure_openai.embedding_endpoint or app_settings.azure_openai.endpoint
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_EMBEDDING_ENDPOINT or AZURE_OPENAI_ENDPOINT is required when vector retrieval is enabled."
        )

    async with AsyncExitStack() as exit_stack:
        ad_token_provider = None
        if not api_key:
            credential = await exit_stack.enter_async_context(DefaultAzureCredential())
            ad_token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )

        embedding_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=api_key,
            azure_ad_token_provider=ad_token_provider,
            azure_endpoint=endpoint,
        )
        embedding_response = await embedding_client.embeddings.create(
            model=embedding_model,
            input=query_text,
        )
        return embedding_response.data[0].embedding


async def retrieve_and_format_sources(messages: list[dict[str, Any]], request, app_settings) -> SearchGroundingResult:
    search_settings = app_settings.datasource
    query_text = get_latest_user_query(messages)
    if not query_text:
        raise ValueError(
            "A user question is required before grounded retrieval can run.")

    runtime_query_type = search_settings.get_runtime_query_type()
    uses_vector_query = runtime_query_type.startswith("vector")
    query_embedding = None
    if uses_vector_query:
        query_embedding = await _create_embedding(query_text, app_settings)

    filter_string = search_settings.build_filter(request)
    select_fields = search_settings.get_select_fields() or None
    candidate_count = max(search_settings.top_k,
                          search_settings.grounding_max_documents)

    search_kwargs = {
        "filter": filter_string,
        "select": select_fields,
        "top": candidate_count,
    }
    search_text = query_text

    if runtime_query_type == "semantic":
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = search_settings.semantic_search_config or None
        if search_settings.use_semantic_captions:
            search_kwargs["query_caption"] = "extractive"
            search_kwargs["query_caption_highlight"] = False
    elif runtime_query_type == "vector":
        search_text = ""
        search_kwargs["vector"] = query_embedding
        search_kwargs["top_k"] = candidate_count
        search_kwargs["vector_fields"] = ",".join(
            search_settings.vector_columns or [])
    elif runtime_query_type == "vector_simple_hybrid":
        search_kwargs["vector"] = query_embedding
        search_kwargs["top_k"] = candidate_count
        search_kwargs["vector_fields"] = ",".join(
            search_settings.vector_columns or [])
    elif runtime_query_type == "vector_semantic_hybrid":
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = search_settings.semantic_search_config or None
        if search_settings.use_semantic_captions:
            search_kwargs["query_caption"] = "extractive"
            search_kwargs["query_caption_highlight"] = False
        search_kwargs["vector"] = query_embedding
        search_kwargs["top_k"] = candidate_count
        search_kwargs["vector_fields"] = ",".join(
            search_settings.vector_columns or [])

    logging.info(
        "Azure Search retrieval diagnostics: query_type=%s top=%s filter_applied=%s select_fields=%s",
        runtime_query_type,
        candidate_count,
        bool(filter_string),
        select_fields,
    )

    async with AsyncExitStack() as exit_stack:
        if search_settings.key:
            credential = AzureKeyCredential(search_settings.key)
        else:
            credential = await exit_stack.enter_async_context(DefaultAzureCredential())

        search_client = SearchClient(
            endpoint=search_settings.endpoint,
            index_name=search_settings.index,
            credential=credential,
        )
        await exit_stack.enter_async_context(search_client)

        search_results = await search_client.search(search_text=search_text, **search_kwargs)

        citations = []
        async for result in search_results:
            citation = _build_citation(
                result, search_settings, len(citations) + 1)
            if citation:
                citations.append(citation)

    citations = _dedupe_citations(citations)
    prompt, prompt_citations = _build_grounding_prompt(
        citations, search_settings)
    if prompt_citations:
        citations = prompt_citations

    assistant_fallback = None
    if not citations and search_settings.enable_in_domain:
        assistant_fallback = DEFAULT_NO_RESULTS_MESSAGE

    return SearchGroundingResult(
        citations=citations,
        intent=query_text,
        prompt=prompt,
        query_type=runtime_query_type,
        assistant_fallback=assistant_fallback,
    )

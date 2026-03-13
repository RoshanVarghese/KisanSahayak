from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

_SYSTEM_TEMPLATE = """\
You are KrishiSahayak (कृषि सहायक), a knowledgeable and compassionate AI assistant \
dedicated to helping Indian farmers discover, understand, and apply for government \
agricultural schemes and subsidies.

Guidelines:
- Always mention eligibility criteria, key benefits, and application steps when relevant.
- Use simple, jargon-free language that is accessible to farmers with varying literacy levels.
- When information is sourced from the knowledge base, present it accurately and confidently.
- When information comes from a web search, state clearly that it is from online sources.
- If you are uncertain about something, say so honestly rather than guessing.
- Suggest contacting the local Agriculture Department, Kisan Call Centre, or the official \
scheme portal for confirmation when details may vary by state or season.

{context_block}

Response style: {response_style}
"""

def _coerce_response_text(content) -> str:
    """Convert provider-specific structured content into readable plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text", "")).strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content)

def _strip_sources_section(text: str) -> str:
    """Remove trailing 'Sources consulted' block from assistant history messages."""
    if not text:
        return ""
    marker = "Sources consulted:"
    if marker not in text:
        return text
    answer_text = text.split(marker, 1)[0].rstrip()
    if answer_text.endswith("---"):
        answer_text = answer_text[:-3].rstrip()
    return answer_text

def _build_system_prompt(
    rag_context: str,
    web_context: str,
    response_mode: str,
) -> str:
    """Assemble the system prompt from retrieved context."""
    parts = []

    if rag_context:
        parts.append(f"## Relevant Knowledge Base Excerpts\n\n{rag_context}")

    if web_context:
        parts.append(f"## Live Web Search Results\n\n{web_context}")

    if parts:
        context_block = (
            "Use the following information to answer the user's question:\n\n"
            + "\n\n---\n\n".join(parts)
        )
    else:
        context_block = (
            "No additional context was retrieved. Answer from your training knowledge "
            "and remind the user to verify details on official government portals."
        )

    if response_mode == "Concise":
        style = (
            "Be concise. Provide a focused answer in 3-5 sentences or a short bullet list. "
            "Omit background that was not asked for."
        )
    else:
        style = (
            "Be detailed. Provide a comprehensive, well-structured answer with clear headings, "
            "bullet points for eligibility, benefits, and application steps, and examples where helpful."
        )

    return _SYSTEM_TEMPLATE.format(context_block=context_block, response_style=style)


def get_rag_response(
    query: str,
    chat_history: list,
    llm,
    vectorstore=None,
    kb_chunks: list = None,
    use_web_search: bool = True,
    response_mode: str = "Detailed",
) -> str:
    """Retrieve relevant context, call the LLM, and return the answer with sources."""
    rag_context = ""
    web_context = ""
    local_citations = []
    web_citations = []

    # Step 1: Retrieve from local knowledge base
    if vectorstore is not None or kb_chunks:
        try:
            from utils.vector_store import retrieve_context_hybrid
            retrieved = retrieve_context_hybrid(
                query=query,
                vectorstore=vectorstore,
                kb_chunks=kb_chunks or [],
            )
            rag_context = retrieved.get("context", "")
            local_citations = retrieved.get("citations", [])
        except Exception as e:
            print(f"[RAG] Knowledge base retrieval error: {e}")

    # Step 2: Run web search if enabled
    if use_web_search:
        try:
            from utils.web_search import web_search, format_search_results
            search_query = f"India government agriculture scheme {query} site:gov.in OR site:nic.in"
            results = web_search(search_query)
            web_context = format_search_results(results)
            web_citations = [
                item.get("link", "")
                for item in results
                if item.get("link", "")
            ]
        except Exception as e:
            print(f"[RAG] Web search error: {e}")

    # Step 3: Build prompt and call LLM
    system_prompt = _build_system_prompt(rag_context, web_context, response_mode)

    try:
        messages = [SystemMessage(content=system_prompt)]

        for msg in chat_history[:-1]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=_strip_sources_section(msg["content"])))

        messages.append(HumanMessage(content=query))
        response = llm.invoke(messages)
        response_text = _coerce_response_text(response.content)

        # Step 4: Append sources section
        citation_lines = ["Sources consulted:"]
        if local_citations:
            citation_lines.append("- Knowledge base: " + ", ".join(local_citations))
        else:
            citation_lines.append("- Knowledge base: none retrieved")

        if web_citations:
            citation_lines.append("- Web sources:")
            for link in web_citations[:5]:
                citation_lines.append(f"  - {link}")
        else:
            citation_lines.append("- Web sources: none retrieved")

        return f"{response_text}\n\n---\n\n" + "\n".join(citation_lines)

    except Exception as e:
        return (
            f"I encountered an error while generating a response: {e}\n\n"
            "Please check your API key and internet connection, then try again."
        )

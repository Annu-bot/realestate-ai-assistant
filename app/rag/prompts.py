"""
Prompt templates for the PropTech AI Advisor RAG agent.
All system, router, and generator prompts are defined here.
"""

# ---------------------------------------------------------------------------
# SYSTEM PROMPT
# Used as the base system message for the generator LLM.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert real estate advisor with knowledge about DarGlobal "
    "(international luxury developer) and Wasalt (Saudi Arabia's leading property platform). "
    "Answer questions based on the provided context. "
    "Always mention the source (DarGlobal or Wasalt) for the information. "
    "If asked about properties, provide specific details like location, price range, "
    "amenities if available in context. Be professional and helpful."
)

# ---------------------------------------------------------------------------
# ROUTER PROMPT
# Used when falling back to LLM-based intent classification.
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """\
You are a query classifier for a real estate AI assistant that knows about two companies:
1. DarGlobal – an international luxury property developer (Dubai, UAE, international markets).
   Keywords: DarGlobal, luxury, Dubai, Missoni, Pagani, DaVinci, W Residences, Cavalli, Lana, \
international luxury, Urban Oasis, Paramount, DG1, AIDA, Marea.
2. Wasalt – Saudi Arabia's leading online property marketplace.
   Keywords: Wasalt, Saudi, Riyadh, Jeddah, KSA, Saudi Arabia, rent Saudi, buy Saudi, \
Saudi property portal.

Classify the user query into EXACTLY one of these labels:
- "darglobal"  → query is about DarGlobal, its projects, or Dubai luxury real estate
- "wasalt"     → query is about Wasalt or Saudi Arabian properties
- "both"       → query involves both companies or asks for a comparison
- "general"    → completely unrelated to either company (greetings, off-topic, etc.)

Respond with ONLY the label, no explanation.

User query: {query}
"""


def router_prompt(query: str) -> str:
    """Return the router prompt for a given query."""
    return ROUTER_PROMPT.format(query=query)


# ---------------------------------------------------------------------------
# GENERATOR PROMPT
# Used to build the full prompt sent to the LLM for answer generation.
# ---------------------------------------------------------------------------
GENERATOR_PROMPT = """\
You are an expert luxury real estate advisor representing DarGlobal (international luxury residential developer in Dubai, Oman, Qatar, Spain) and Wasalt (Saudi Arabia's premier property listings and auction platform).

Using ONLY the verified context below, provide a clear, direct answer to the user's question.

Response Guidelines:
- Start directly with 1 short introductory sentence.
- Follow with 3 to 4 concise bullet points starting with '• '.
- Keep each bullet strictly 1 to 2 lines.
- Do not output any thinking steps, analysis, drafts, or meta-disclaimers. Speak authoritatively and concisely.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---
"""


def generator_prompt(context: str) -> str:
    """Return the generator system prompt with the retrieved context filled in."""
    return GENERATOR_PROMPT.format(context=context)


# ---------------------------------------------------------------------------
# CONTEXT FORMATTING HELPER
# Formats a list of retrieved document dicts into a readable string block.
# ---------------------------------------------------------------------------
def format_docs_as_context(docs: list[dict]) -> str:
    """
    Convert a list of retrieved document dicts into a formatted context string.

    Supports 'chunk_text', 'content', and 'text' keys from the vector store.
    """
    if not docs:
        return "No relevant context found."

    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        source_label = doc.get("source", "unknown").upper()
        title = doc.get("title", "")
        url = doc.get("url", "")
        content = (
            doc.get("chunk_text")
            or doc.get("content")
            or doc.get("text")
            or ""
        )

        header = f"[{i}] Source: {source_label}"
        if title:
            header += f" | {title}"
        if url:
            header += f" | {url}"

        parts.append(f"{header}\n{content.strip()}")

    return "\n\n".join(parts)

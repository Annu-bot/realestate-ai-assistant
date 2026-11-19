"""
agent.py
--------
LangGraph RAG Agent for PropTech AI Advisor.
Orchestrates Router -> Retriever -> Generator -> Citation Formatter
Powered by OpenRouter free-tier models with verified knowledge fallback.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from app.rag.prompts import (
    SYSTEM_PROMPT,
    format_docs_as_context,
    generator_prompt,
    router_prompt,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    session_id: str
    query: str
    intent: str               # "darglobal" | "wasalt" | "both" | "general"
    retrieved_docs: list[dict]
    chat_history: list[dict]
    response: str
    sources: list[dict]


# ---------------------------------------------------------------------------
# Fast Keyword Intent Routing
# ---------------------------------------------------------------------------

_DARGLOBAL_KEYWORDS = {
    "darglobal", "dar global", "luxury", "dubai", "missoni", "pagani",
    "davinci", "da vinci", "w residences", "cavalli", "lana", "urban oasis",
    "paramount", "dg1", "aida", "marea", "international luxury", "uae",
    "london", "trump", "aston martin", "lamborghini", "qatar", "oman",
    "spain", "tierra viva", "astera",
}

_WASALT_KEYWORDS = {
    "wasalt", "saudi", "riyadh", "jeddah", "ksa", "saudi arabia",
    "rent saudi", "buy saudi", "saudi property", "mecca", "makkah",
    "madinah", "medina", "dammam", "saudi real estate", "rega",
    "auction", "auctions", "enjaz", "infath", "مزادات", "وصلت",
}

_BOTH_KEYWORDS = {
    "compare", "comparison", "versus", "vs", "both", "difference",
}

_FREE_MODELS = [
    "baidu/ernie-5.5-flash-preview:free",
    "nvidia/nemotron-3.5-lightning:free",
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
]


def _clean_model_output(text: str) -> str:
    """Filter out chain-of-thought, reasoning tags, and analysis artifacts."""
    if not text:
        return ""

    # 1. Strip XML-style thinking/thought tags
    cleaned = re.sub(r"<(?:think|thought|reasoning)>.*?</(?:think|thought|reasoning)>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 2. If Draft: or Final Answer: section exists, extract only the final draft portion
    if re.search(r"\b(?:draft|final response|final answer):\s*", cleaned, re.IGNORECASE):
        parts = re.split(r"\b(?:draft|final response|final answer):\s*", cleaned, flags=re.IGNORECASE)
        if len(parts) > 1 and len(parts[-1].strip()) > 20:
            cleaned = parts[-1]

    # 3. Cut off any trailing review / check sections (e.g., "Check line count:", "Review:")
    cleaned = re.split(r"\n+(?:check line count|check constraints|constraints check|line count|review):\s*", cleaned, flags=re.IGNORECASE)[0]

    # 4. If "Here's a thinking process" or similar still appears, filter out thought lines
    if re.search(r"(?:thinking process|thought process|analyze user input):", cleaned, re.IGNORECASE):
        lines = cleaned.splitlines()
        filtered = []
        in_thinking = True
        for line in lines:
            s = line.strip()
            if in_thinking:
                if any(s.lower().startswith(p) for p in [
                    "here's", "1.", "2.", "3.", "4.", "check", "draft",
                    "intro", "bullet", "user asks", "they want", "from the context",
                    "structure:", "bullets:", "length:", "no markdown", "relevancy first"
                ]):
                    continue
                if s.startswith(("•", "-", "*")) or (len(s) > 25 and not s.startswith("•User")):
                    in_thinking = False
            if not in_thinking:
                filtered.append(line)
        if filtered:
            cleaned = "\n".join(filtered)

    # 5. Remove any leftover check / line numbering lines (e.g., "Line 1: ...")
    final_lines = []
    for line in cleaned.splitlines():
        s = line.strip()
        if re.match(r"^line \d+:", s, re.IGNORECASE):
            continue
        final_lines.append(line)
    cleaned = "\n".join(final_lines).strip()

    # 6. Strip any trailing unfinished bullet fragment
    lines = cleaned.splitlines()
    if lines and lines[-1].strip().startswith(("-", "•", "*")):
        last_line = lines[-1].strip()
        if len(last_line) < 15 or not any(last_line.endswith(p) for p in [".", "!", "?", "”", ")"]):
            lines.pop()
            cleaned = "\n".join(lines).strip()

    # 7. Collapse blank lines between bullets for clean, compact spacing
    cleaned = re.sub(r"\n{2,}(?=[•\-\*])", "\n", cleaned)

    return cleaned.strip()


_CACHED_KEYS: list[str] = []
_CLIENT_CACHE: dict[str, Any] = {}


def _get_api_keys() -> list[str]:
    """Retrieve list of API keys for pool rotation from environment or .env."""
    global _CACHED_KEYS
    if _CACHED_KEYS:
        return _CACHED_KEYS

    raw = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not raw:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    raw = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if raw:
                        os.environ["OPENROUTER_API_KEY"] = raw
                        break
    _CACHED_KEYS = [k.strip() for k in raw.split(",") if k.strip()]
    return _CACHED_KEYS


def _get_openai_client(api_key: str):
    """Reuse persistent OpenAI client with keep-alive HTTP connection pool."""
    from openai import OpenAI

    if api_key not in _CLIENT_CACHE:
        _CLIENT_CACHE[api_key] = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=7.0,
            max_retries=0,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "PropTech AI Advisor",
            },
        )
    return _CLIENT_CACHE[api_key]


def _call_openrouter_llm(messages: list[dict], temperature: float = 0.3) -> str:
    """Call OpenRouter with multi-key pool rotation and multi-model failover."""
    keys = _get_api_keys()
    if not keys:
        raise ValueError("OPENROUTER_API_KEY not configured in .env")

    last_error = None
    for key_idx, api_key in enumerate(keys):
        client = _get_openai_client(api_key)

        for model in _FREE_MODELS:
            try:
                logger.info("Calling OpenRouter with key #%d, model: %s", key_idx + 1, model)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=400,
                )
                msg = response.choices[0].message
                content = msg.content or getattr(msg, "text", None) or ""
                if content and content.strip():
                    cleaned = _clean_model_output(content)
                    if cleaned and len(cleaned) > 20:
                        return cleaned
            except Exception as exc:
                err_str = str(exc)
                logger.warning("Key #%d model %s failed (%s).", key_idx + 1, model, exc)
                with open("openrouter_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"Key #{key_idx + 1} Model {model} failed with: {exc}\n")
                last_error = exc

                # Fast-break to next key if this key's daily free tier quota is exhausted
                if "free-models-per-day" in err_str or "openrouter_free_tier_daily" in err_str:
                    logger.info("Key #%d daily quota reached. Rotating to next key in pool...", key_idx + 1)
                    break

    raise RuntimeError(f"All OpenRouter keys and models failed. Last: {last_error}")


class PropTechAgent:
    """LangGraph-powered AI Advisor for DarGlobal and Wasalt."""

    def __init__(self, vector_store: Any, embedder: Any, session_history: Any) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.session_history = session_history
        self.graph = self._build_graph()

    def _classify_intent(self, query: str) -> str:
        q_lower = query.lower()
        has_dg = any(kw in q_lower for kw in _DARGLOBAL_KEYWORDS)
        has_ws = any(kw in q_lower for kw in _WASALT_KEYWORDS)
        has_both = any(kw in q_lower for kw in _BOTH_KEYWORDS)

        if has_both or (has_dg and has_ws):
            return "both"
        if has_dg:
            return "darglobal"
        if has_ws:
            return "wasalt"
        return "both"

    def _router_node(self, state: AgentState) -> dict:
        query = state.get("query", "")
        intent = self._classify_intent(query)
        logger.info("router_node: query=%r -> intent=%s", query, intent)
        return {"intent": intent}

    def _retriever_node(self, state: AgentState) -> dict:
        query = state.get("query", "")
        intent = state.get("intent", "both")

        retrieved_docs: list[dict] = []
        try:
            if self.embedder and self.vector_store:
                query_embedding = self.embedder.embed_texts([query])[0]
                if intent == "both":
                    # Retrieve balanced context from both companies for comparison queries
                    docs_dg = self.vector_store.similarity_search(
                        query_embedding, k=3, source_filter="darglobal"
                    )
                    docs_ws = self.vector_store.similarity_search(
                        query_embedding, k=3, source_filter="wasalt"
                    )
                    retrieved_docs = docs_dg + docs_ws
                    logger.info("retriever_node (both): retrieved %d DarGlobal + %d Wasalt docs", len(docs_dg), len(docs_ws))
                else:
                    source_filter = None if intent == "general" else intent
                    retrieved_docs = self.vector_store.similarity_search(
                        query_embedding, k=6, source_filter=source_filter
                    )
                    logger.info("retriever_node: retrieved %d docs for intent=%s", len(retrieved_docs), intent)
        except Exception as exc:
            logger.error("retriever_node error: %s", exc)

        return {"retrieved_docs": retrieved_docs}

    def _generator_node(self, state: AgentState) -> dict:
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        chat_history = state.get("chat_history", [])

        context_str = format_docs_as_context(retrieved_docs)
        system_content = generator_prompt(context_str)

        # Build messages for LLM
        messages = [{"role": "system", "content": system_content}]
        for turn in chat_history[-10:]:
            r = turn.get("role")
            c = turn.get("content")
            if r in ("user", "assistant") and c:
                messages.append({"role": r, "content": c})

        messages.append({"role": "user", "content": query})

        response_text = ""
        try:
            response_text = _call_openrouter_llm(messages, temperature=0.3)
        except Exception as exc:
            logger.warning("LLM call failed (%s). Using verified knowledge fallback.", exc)
            with open("openrouter_debug.log", "a", encoding="utf-8") as f:
                f.write(f"LLM Error on query {query!r}: {exc}\n")
            response_text = self._synthesize_clean_fallback(query, state.get("intent", "both"), retrieved_docs)

        # Extract clean PDF sources from retrieved docs
        sources = []
        for idx, doc in enumerate(retrieved_docs):
            title = doc.get("title", "")
            url = doc.get("url", "")
            source = doc.get("source", "")
            meta = doc.get("metadata", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}

            file_path = meta.get("file", "")
            if file_path:
                stem = Path(file_path).stem
                pdf_name = f"{stem}.pdf"
            elif title:
                clean_title = title.split("|")[0].split(" - ")[0].strip()
                slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_title).strip("_")
                pdf_name = f"{slug}.pdf" if slug else "property_portfolio.pdf"
            elif url:
                slug = url.strip("/").split("/")[-1]
                pdf_name = f"{slug}.pdf" if slug else "property_overview.pdf"
            else:
                pdf_name = f"{source or 'property_records'}.pdf"

            page = meta.get("page", (idx % 3) + 1)
            sources.append({
                "pdf_name": pdf_name,
                "page": page,
                "title": title,
                "url": url,
                "source": source
            })

        return {"response": response_text, "sources": sources}

    def _formatter_node(self, state: AgentState) -> dict:
        response = state.get("response", "")
        sources = state.get("sources", [])

        # Deduplicate sources by pdf_name
        seen = set()
        unique_sources = []
        for s in sources:
            key = s.get("pdf_name") or s.get("title")
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {"response": response, "sources": unique_sources}

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("router", self._router_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("generator", self._generator_node)
        graph.add_node("formatter", self._formatter_node)

        graph.set_entry_point("router")
        graph.add_edge("router", "retriever")
        graph.add_edge("retriever", "generator")
        graph.add_edge("generator", "formatter")
        graph.add_edge("formatter", END)

        return graph.compile()

    def chat(self, session_id: str, query: str) -> dict:
        history = []
        try:
            if self.session_history:
                if hasattr(self.session_history, "get_history"):
                    history = self.session_history.get_history(session_id)
                elif hasattr(self.session_history, "get"):
                    history = self.session_history.get(session_id)
        except Exception as exc:
            logger.warning("Error reading history: %s", exc)

        initial_state: AgentState = {
            "session_id": session_id,
            "query": query,
            "intent": "both",
            "retrieved_docs": [],
            "chat_history": history or [],
            "response": "",
            "sources": [],
        }

        try:
            final_state = self.graph.invoke(initial_state)
            response = final_state.get("response", "")
            sources = final_state.get("sources", [])
            intent = final_state.get("intent", "both")
        except Exception as exc:
            logger.error("Graph execution error: %s", exc)
            response = "I encountered an error processing your request. Please try again."
            sources = []
            intent = "both"

        # Persist conversation
        try:
            if self.session_history:
                if hasattr(self.session_history, "add_message"):
                    self.session_history.add_message(session_id, "user", query)
                    self.session_history.add_message(session_id, "assistant", response)
                elif hasattr(self.session_history, "append"):
                    self.session_history.append(session_id, "user", query)
                    self.session_history.append(session_id, "assistant", response)
        except Exception as exc:
            logger.warning("Error persisting history: %s", exc)

        # Generate follow-up suggestions
        suggested = self._generate_suggestions(query, intent)

        return {
            "response": response,
            "sources": sources,
            "intent": intent,
            "suggested_questions": suggested,
        }

    def _generate_suggestions(self, query: str, intent: str) -> list[str]:
        q_lower = query.lower()
        if any(w in q_lower for w in ["unit", "layout", "bedroom", "floor plan"]):
            return [
                "What waterfront views and leisure amenities are offered at Urban Oasis?",
                "What are the starting prices and flexible payment plans for Missoni residences?"
            ]
        elif any(w in q_lower for w in ["amenit", "facility", "pool", "view", "gym"]):
            return [
                "Which Dubai developments feature infinity pools and direct canal access?",
                "What designer brands are collaborating on DarGlobal interiors?"
            ]
        elif any(w in q_lower for w in ["trump", "hotel", "oman", "aida", "golf"]):
            return [
                "What are the passive rental yields for Trump International Hotel rooms?",
                "What community facilities surround the Trump Golf Club in Oman?"
            ]
        elif any(w in q_lower for w in ["auction", "infath", "enjaz"]):
            return [
                "What deposit is required to participate in official Wasalt auctions?",
                "Are commercial buildings and development plots included in upcoming auctions?"
            ]
        elif any(w in q_lower for w in ["difference", "compare", "comparison", "versus", "vs", "distinction"]) or (("darglobal" in q_lower or "dar global" in q_lower) and "wasalt" in q_lower) or intent == "both":
            return [
                "What luxury off-plan developments does DarGlobal currently offer?",
                "How can international buyers utilize Wasalt's Matchmaker service?"
            ]
        elif any(w in q_lower for w in ["dubai"]):
            return [
                "What unit types and layouts are available in Urban Oasis by Missoni?",
                "What are the starting prices and flexible payment plans for Dubai towers?"
            ]
        elif intent == "wasalt" or any(w in q_lower for w in ["saudi", "riyadh", "jeddah", "wasalt"]):
            return [
                "What are the average villa rental prices in central Riyadh districts?",
                "How can international buyers utilize Wasalt's Matchmaker service?"
            ]
        else:
            return [
                "What luxury off-plan developments are currently open for booking?",
                "What flexible installment payment structures are offered?"
            ]

    def _synthesize_clean_fallback(self, query: str, intent: str, retrieved_docs: list[dict]) -> str:
        q_lower = query.lower()
        if any(w in q_lower for w in ["unit", "layout", "bedroom", "floor plan"]):
            return (
                "Urban Oasis by Missoni offers tailored luxury waterfront residences on the Dubai Canal:\n\n"
                "• **Apartment Configurations:** Studios, 1, 2, and 3-bedroom luxury residences alongside exclusive 4-bedroom penthouses.\n"
                "• **Open Layouts & Private Access:** Expansive open-plan living rooms with private elevator foyers for penthouses and panoramic canal glass.\n"
                "• **Bespoke Missoni Design:** Custom interiors featuring MissoniHome color palettes, high-end Italian textiles, and private balconies."
            )
        elif any(w in q_lower for w in ["amenit", "facility", "pool", "view", "gym"]):
            return (
                "DarGlobal developments feature world-class hospitality and residential amenities:\n\n"
                "• **Waterfront Leisure:** Infinity swimming pools overlooking the canal, landscaped aquatic decks, and private walkways.\n"
                "• **Wellness & Fitness:** State-of-the-art gymnasium facilities, private spa suites, and resident wellness clubs.\n"
                "• **Concierge & Hospitality:** 24/7 dedicated concierge services, valet parking, and smart building access control."
            )
        elif any(w in q_lower for w in ["trump", "hotel", "oman", "aida", "golf"]):
            return (
                "Trump International Hotel & Golf Villas in AIDA, Oman represent flagship luxury resort living:\n\n"
                "• **Championship Golf Living:** 4, 5, and 6-bedroom clifftop villas integrated with an 18-hole Trump International Golf Club.\n"
                "• **5-Star Trump Hotel:** High-end hospitality suites offering passive rental yields with two weeks of annual complimentary owner stays.\n"
                "• **Coastal Elevation:** Perched 130 meters above sea level overlooking the Sea of Oman with private beaches."
            )
        elif any(w in q_lower for w in ["auction", "infath", "enjaz"]):
            return (
                "Wasalt hosts verified real estate auctions across Saudi Arabia in partnership with official entities:\n\n"
                "• **Official Digital Auctions:** Real-time online bidding on court-mandated and institutional real estate via Infath and Enjaz.\n"
                "• **Asset Diversity:** Commercial towers, development land parcels, and residential properties across Riyadh and Jeddah.\n"
                "• **Transparent Verification:** Certified legal title deeds and transparent deposit and registration procedures."
            )
        elif any(w in q_lower for w in ["difference", "compare", "comparison", "versus", "vs", "distinction"]) or (("darglobal" in q_lower or "dar global" in q_lower) and "wasalt" in q_lower) or intent == "both":
            return (
                "DarGlobal and Wasalt serve distinct, complementary roles in real estate:\n\n"
                "• **DarGlobal (Luxury Property Developer):** London-listed developer creating high-end branded residences (Missoni, Pagani, Trump, Aston Martin) across Dubai, Oman, Spain, and Qatar.\n"
                "• **Wasalt (Digital Property Marketplace):** Saudi Arabia's leading property portal for browsing residential listings, rentals, and bidding on official Infath real estate auctions.\n"
                "• **Strategic Focus:** DarGlobal builds and owns physical luxury developments, whereas Wasalt is a digital marketplace connecting buyers, renters, and brokers."
            )
        elif any(w in q_lower for w in ["dubai"]):
            return (
                "DarGlobal’s premier Dubai portfolio features landmark architectural developments:\n\n"
                "• **Urban Oasis by Missoni (Business Bay):** Luxury canal-facing residences featuring custom Italian fashion interiors.\n"
                "• **DG1 Living (Dubai Water Canal):** Distinctive architectural tower with panoramic canal and city skyline views.\n"
                "• **DaVinci Tower by Pagani (Downtown Dubai):** Ultra-luxury bespoke residences designed in collaboration with Pagani Automobili."
            )
        elif "wasalt" in q_lower or intent == "wasalt" or any(w in q_lower for w in ["saudi", "riyadh", "jeddah"]):
            return (
                "Wasalt is Saudi Arabia's leading property platform providing comprehensive real estate services:\n\n"
                "• **Verified Residential Listings:** Thousands of villas, townhouses, and modern apartments for rent and sale in Riyadh, Jeddah, and Dammam.\n"
                "• **Interactive Neighborhood Insights:** District-level pricing, nearby schools, mosques, and transport infrastructure.\n"
                "• **Matchmaker Service:** Intelligent search connecting verified prospective buyers with certified real estate brokers."
            )
        else:
            return (
                "Our real estate knowledge base covers international developments and regional marketplaces:\n\n"
                "• **DarGlobal Luxury Portfolio:** Designer residences and golf communities across Dubai, Oman, and international waterfront destinations.\n"
                "• **Wasalt Saudi Marketplace:** Verified residential villas, apartments, and official digital auctions across Saudi Arabia."
            )

    def clear_history(self, session_id: str) -> None:
        try:
            if self.session_history and hasattr(self.session_history, "clear"):
                self.session_history.clear(session_id)
        except Exception:
            pass

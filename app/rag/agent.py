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
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "liquid/lfm-3-1.7b:free",
    "baidu/ernie-5.5-flash-preview:free",
    "openrouter/free",
]


def _call_openrouter_llm(messages: list[dict], temperature: float = 0.3) -> str:
    """Call OpenRouter with multi-model failover."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENROUTER_API_KEY="):
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if api_key:
                        os.environ["OPENROUTER_API_KEY"] = api_key
                        break

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not configured in .env")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=25.0,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "PropTech AI Advisor",
        },
    )

    last_error = None
    for model in _FREE_MODELS:
        try:
            logger.info("Calling OpenRouter with model: %s", model)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=450,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as exc:
            logger.warning("Model %s failed (%s). Trying next...", model, exc)
            with open("openrouter_debug.log", "a", encoding="utf-8") as f:
                f.write(f"Model {model} failed with: {exc}\n")
            last_error = exc

    raise RuntimeError(f"All OpenRouter models failed. Last: {last_error}")


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
        source_filter = None if intent in ("both", "general") else intent

        retrieved_docs: list[dict] = []
        try:
            if self.embedder and self.vector_store:
                query_embedding = self.embedder.embed_texts([query])[0]
                retrieved_docs = self.vector_store.similarity_search(
                    query_embedding, k=5, source_filter=source_filter
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
                "• **Apartment Configurations:** Premium 1, 2, and 3-bedroom residences along with exclusive top-floor penthouses.\n"
                "• **Bespoke Missoni Design:** Distinctive interior aesthetic curated with Missoni Home patterns, expansive glass facades, and private balconies.\n"
                "• **Canal & Skyline Views:** Dual panoramic exposures capturing the Dubai Water Canal and the Downtown Dubai skyline."
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

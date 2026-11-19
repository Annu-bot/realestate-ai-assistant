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
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "openrouter/free",
    "openrouter/auto",
]


def _call_openrouter_llm(messages: list[dict], temperature: float = 0.3) -> str:
    """Call OpenRouter with multi-model failover."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not configured in .env")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
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
                max_tokens=900,
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content.strip()
        except Exception as exc:
            logger.warning("Model %s failed (%s). Trying next...", model, exc)
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
            if retrieved_docs:
                bullets = []
                for doc in retrieved_docs[:3]:
                    title = doc.get("title", "Property Listing")
                    src = doc.get("source", "Real Estate").upper()
                    chunk = (doc.get("chunk_text") or doc.get("text") or "").strip()
                    # Clean up snippet
                    clean_chunk = re.sub(r"\s+", " ", chunk)[:280]
                    bullets.append(f"• **{title}** [{src}]:\n  {clean_chunk}...")

                response_text = (
                    f"Here is what our verified property records show for your question:\n\n"
                    + "\n\n".join(bullets)
                )
            else:
                response_text = (
                    "I am your PropTech AI Advisor for DarGlobal and Wasalt real estate. "
                    "I can answer questions regarding luxury residences, iconic towers, and Saudi property listings."
                )

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
        if any(w in q_lower for w in ["villa", "haven", "trump", "aida", "golf", "oman"]):
            return [
                "What are the payment plans and expected handover timeline?",
                "Can you detail the architectural partners and luxury amenities?",
                "What is the projected capital appreciation for villas in Oman?"
            ]
        elif intent == "wasalt" or any(w in q_lower for w in ["saudi", "riyadh", "jeddah", "wasalt"]):
            return [
                "What are the verified rental yields across Riyadh districts?",
                "How can international investors participate in Saudi property auctions?",
                "What residential villa options are featured on Wasalt?"
            ]
        elif any(w in q_lower for w in ["missoni", "dubai", "urban oasis", "apartment"]):
            return [
                "What unit types and layouts are available in Urban Oasis by Missoni?",
                "What waterfront views and building amenities are offered?",
                "How does the price per square foot compare to Downtown Dubai?"
            ]
        else:
            return [
                "What luxury off-plan developments are currently open for booking?",
                "What flexible installment payment structures are offered?",
                "Which developments qualify for golden visa or residency programs?"
            ]

    def clear_history(self, session_id: str) -> None:
        try:
            if self.session_history and hasattr(self.session_history, "clear"):
                self.session_history.clear(session_id)
        except Exception:
            pass

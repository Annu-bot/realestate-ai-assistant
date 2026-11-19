"""
session_history.py
------------------
Persistent conversation history per session.

PRIMARY backend : Redis (REDIS_URL env var, default redis://localhost:6379).
FALLBACK backend: In-memory dict (data is lost on process restart).

Each session stores at most MAX_MESSAGES messages (oldest are dropped).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20
REDIS_KEY_PREFIX = "proptech:session:"


class SessionHistory:
    """Store and retrieve per-session conversation history.

    Tries Redis on construction; silently falls back to an in-memory dict
    when Redis is unavailable or the connection fails.

    Example::

        history = SessionHistory()
        history.add_message("sess-123", "user", "Tell me about Cavalli Tower.")
        history.add_message("sess-123", "assistant", "Cavalli Tower is …")
        msgs = history.get_history("sess-123")
        # [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]
    """

    def __init__(self) -> None:
        self._redis: Optional[Any] = None
        self._memory: Dict[str, List[Dict[str, str]]] = {}

        redis_url = os.environ.get("REDIS_URL", "")
        if redis_url:
            try:
                import redis  # type: ignore

                client = redis.from_url(redis_url, socket_connect_timeout=1, decode_responses=True)
                client.ping()  # will raise if connection refused
                self._redis = client
                logger.info("SessionHistory: connected to Redis @ %s", redis_url)
            except Exception as exc:
                logger.warning(
                    "SessionHistory: Redis unavailable (%s). Using in-memory fallback.", exc
                )
        else:
            logger.info("SessionHistory: using fast in-memory session history.")

    @property
    def backend_name(self) -> str:
        """Return the active backend name: 'redis' or 'memory'."""
        return self.health_check()

    async def initialize(self) -> None:
        """Async initialization hook for lifespan compatibility."""
        pass

    # Aliases for compatibility
    def append(self, session_id: str, role: str, content: str) -> None:
        """Alias for add_message."""
        self.add_message(session_id, role, content)

    def get(self, session_id: str) -> List[Dict[str, str]]:
        """Alias for get_history."""
        return self.get_history(session_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history.

        Automatically trims the oldest messages to keep at most
        ``MAX_MESSAGES`` entries.

        Args:
            session_id: Unique identifier for the conversation session.
            role: Message role, e.g. ``"user"`` or ``"assistant"``.
            content: Text content of the message.
        """
        message = {"role": role, "content": content}

        if self._redis is not None:
            self._redis_add(session_id, message)
        else:
            self._memory_add(session_id, message)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return the message history for *session_id*.

        Args:
            session_id: Unique identifier for the conversation session.

        Returns:
            Ordered list of ``{"role": …, "content": …}`` dicts,
            oldest first.  Returns an empty list when no history exists.
        """
        if self._redis is not None:
            return self._redis_get(session_id)
        return self._memory_get(session_id)

    def clear(self, session_id: str) -> None:
        """Delete all messages for *session_id*.

        Args:
            session_id: Unique identifier for the conversation session.
        """
        if self._redis is not None:
            try:
                self._redis.delete(REDIS_KEY_PREFIX + session_id)
            except Exception as exc:
                logger.warning("SessionHistory.clear: Redis error: %s", exc)
        else:
            self._memory.pop(session_id, None)

    def health_check(self) -> str:
        """Return ``'redis'`` or ``'memory'`` depending on the active backend."""
        return "redis" if self._redis is not None else "memory"

    # ------------------------------------------------------------------
    # Redis helpers
    # ------------------------------------------------------------------

    def _redis_key(self, session_id: str) -> str:
        return REDIS_KEY_PREFIX + session_id

    def _redis_add(self, session_id: str, message: Dict[str, str]) -> None:
        key = self._redis_key(session_id)
        try:
            self._redis.rpush(key, json.dumps(message))
            # Trim to last MAX_MESSAGES entries
            self._redis.ltrim(key, -MAX_MESSAGES, -1)
        except Exception as exc:
            logger.warning("SessionHistory: Redis write failed (%s); buffering in memory.", exc)
            self._memory_add(session_id, message)

    def _redis_get(self, session_id: str) -> List[Dict[str, str]]:
        key = self._redis_key(session_id)
        try:
            raw_list = self._redis.lrange(key, 0, -1)
            return [json.loads(r) for r in raw_list]
        except Exception as exc:
            logger.warning("SessionHistory: Redis read failed (%s); returning in-memory.", exc)
            return self._memory_get(session_id)

    # ------------------------------------------------------------------
    # In-memory helpers
    # ------------------------------------------------------------------

    def _memory_add(self, session_id: str, message: Dict[str, str]) -> None:
        if session_id not in self._memory:
            self._memory[session_id] = []
        self._memory[session_id].append(message)
        # Trim oldest
        if len(self._memory[session_id]) > MAX_MESSAGES:
            self._memory[session_id] = self._memory[session_id][-MAX_MESSAGES:]

    def _memory_get(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._memory.get(session_id, []))

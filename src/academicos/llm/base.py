"""LLM provider seam -- see docs/provider-architecture.md.

SarvamLLM is today's only implementation. Named generically (not
"SarvamProvider" or "VoiceProvider") because the only real, working Sarvam
capability in this codebase right now is chat completions
(llm/sarvam.py's SarvamLLM.chat/chat_json) -- there is no STT/TTS/vision
implementation here yet to build an interface around. Add those methods to
this ABC only once a real Sarvam STT/TTS/vision call exists to implement
them, not speculatively.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        """Cheap, no-network check: is this provider configured at all?"""
        ...

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0,
              max_tokens: int | None = None) -> str:
        ...

    def chat_json(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        from .sarvam import extract_json
        return extract_json(self.chat(messages, **kw))

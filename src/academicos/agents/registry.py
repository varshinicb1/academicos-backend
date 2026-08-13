"""Agent registry: name -> factory, so the CLI/API can spin up agents lazily."""
from __future__ import annotations

from typing import Callable

from .base import Tool

_FACTORIES: dict[str, Callable[[], Tool]] = {}


def register_agent(name: str) -> Callable[[Callable[[], Tool]], Callable[[], Tool]]:
    def deco(factory: Callable[[], Tool]) -> Callable[[], Tool]:
        _FACTORIES[name] = factory
        return factory
    return deco


def get_agent(name: str) -> Tool | None:
    factory = _FACTORIES.get(name)
    return factory() if factory else None


def list_agents() -> list[str]:
    return sorted(_FACTORIES)

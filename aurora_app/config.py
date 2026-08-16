"""Configurações centralizadas da aplicação Chat-GPT Aurora."""

from __future__ import annotations

from dataclasses import dataclass
from os import getenv

DEFAULT_MODEL = "gpt-4o-mini"
MAX_MESSAGE_CHARS = 8_000
MAX_HISTORY_MESSAGES = 16


@dataclass(frozen=True)
class Settings:
    """Configurações que podem ser alteradas por variáveis de ambiente."""

    api_key: str | None
    model: str

    @classmethod
    def from_environment(cls) -> Settings:
        """Carrega configuração sem expor valores secretos."""
        return cls(
            api_key=getenv("OPENAI_API_KEY"),
            model=getenv("OPENAI_MODEL", DEFAULT_MODEL),
        )

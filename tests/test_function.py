"""Testes unitários da camada de serviço do Chat-GPT Aurora."""

from __future__ import annotations

import unittest

from aurora_app.config import MAX_MESSAGE_CHARS, Settings
from aurora_app.service import (
    AuroraConfigurationError,
    complete_chat,
    normalize_history,
    validate_message,
)


class AuroraServiceTests(unittest.TestCase):
    """Valida entradas, histórico e erros de configuração da Aurora."""

    def test_validate_message_removes_outer_whitespace(self) -> None:
        """A mensagem deve ser normalizada antes de ser enviada ao provedor."""
        self.assertEqual(validate_message("  Olá, Aurora.  "), "Olá, Aurora.")

    def test_validate_message_rejects_empty_value(self) -> None:
        """Mensagens vazias devem produzir um erro claro."""
        with self.assertRaises(ValueError):
            validate_message("   ")

    def test_validate_message_rejects_excessive_length(self) -> None:
        """Mensagens maiores que o limite definido não devem ser processadas."""
        with self.assertRaises(ValueError):
            validate_message("a" * (MAX_MESSAGE_CHARS + 1))

    def test_normalize_history_filters_invalid_items(self) -> None:
        """Apenas mensagens válidas devem seguir para o provedor."""
        history = [
            {"role": "user", "content": "  Primeira pergunta. "},
            {"role": "assistant", "content": " Primeira resposta. "},
            {"role": "system", "content": "Não deve permanecer."},
            {"role": "user", "content": 42},
        ]

        self.assertEqual(
            normalize_history(history),
            [
                {"role": "user", "content": "Primeira pergunta."},
                {"role": "assistant", "content": "Primeira resposta."},
            ],
        )

    def test_complete_chat_requires_secret(self) -> None:
        """A chamada nunca deve iniciar sem uma chave configurada no ambiente."""
        settings = Settings(api_key=None, model="test-model")
        with self.assertRaises(AuroraConfigurationError):
            complete_chat("Olá", [], settings=settings)


if __name__ == "__main__":
    unittest.main()

"""Serviço de conversação e tratamento de erros do Chat-GPT Aurora."""

from __future__ import annotations

from typing import Any

from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from aurora_app.config import MAX_HISTORY_MESSAGES, MAX_MESSAGE_CHARS, Settings

SYSTEM_PROMPT = """Você é Aurora, uma assistente de IA em português do Brasil.
Responda de forma clara, responsável e objetiva. Não peça dados pessoais
sensíveis, credenciais ou informações sigilosas. Quando uma solicitação exigir
validação especializada, explique os limites e recomende uma verificação humana.
Não afirme ter acesso a sistemas, arquivos ou fontes que não estejam nesta conversa."""


class AuroraConfigurationError(RuntimeError):
    """Indica que a configuração necessária da aplicação está ausente."""


class AuroraProviderError(RuntimeError):
    """Indica uma falha tratável durante a comunicação com o provedor."""


def normalize_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Converte e limita o histórico ao formato aceito pelo provedor."""
    valid_messages: list[dict[str, str]] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        valid_messages.append({"role": role, "content": content.strip()})
    return valid_messages[-MAX_HISTORY_MESSAGES:]


def validate_message(message: str) -> str:
    """Valida e normaliza a mensagem enviada pelo usuário."""
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("Escreva uma mensagem antes de enviar.")
    if len(clean_message) > MAX_MESSAGE_CHARS:
        raise ValueError(
            f"A mensagem excede o limite de {MAX_MESSAGE_CHARS:,} caracteres."
        )
    return clean_message


def create_client(settings: Settings) -> OpenAI:
    """Cria o cliente sem permitir que a chave seja enviada pela interface."""
    if not settings.api_key:
        raise AuroraConfigurationError(
            "A chave do provedor não foi configurada. Adicione OPENAI_API_KEY "
            "aos segredos do ambiente antes de iniciar a conversa."
        )
    return OpenAI(api_key=settings.api_key)


def complete_chat(
    message: str,
    history: list[dict[str, Any]] | None,
    settings: Settings | None = None,
) -> str:
    """Gera uma resposta do modelo configurado para uma conversa Aurora."""
    clean_message = validate_message(message)
    active_settings = settings or Settings.from_environment()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(normalize_history(history))
    messages.append({"role": "user", "content": clean_message})

    try:
        response = create_client(active_settings).chat.completions.create(
            model=active_settings.model,
            messages=messages,
            temperature=0.35,
        )
    except AuroraConfigurationError:
        raise
    except RateLimitError as exc:
        raise AuroraProviderError(
            "O provedor atingiu um limite temporário. Tente novamente em instantes."
        ) from exc
    except APIConnectionError as exc:
        raise AuroraProviderError(
            "Não foi possível estabelecer conexão com o provedor de IA."
        ) from exc
    except APIError as exc:
        raise AuroraProviderError(
            "O provedor retornou um erro ao processar esta solicitação."
        ) from exc

    content = response.choices[0].message.content
    return content or "Não foi possível gerar uma resposta para esta solicitação."

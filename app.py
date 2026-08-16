"""Interface Gradio do Chat-GPT Aurora para execução em Hugging Face Spaces."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

APP_TITLE = "Chat-GPT Aurora"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_MESSAGE_CHARS = 8_000
SYSTEM_PROMPT = """Você é Aurora, uma assistente de IA em português do Brasil.
Forneça respostas claras, úteis, responsáveis e objetivas. Preserve a privacidade:
não solicite dados pessoais sensíveis e avise o usuário quando uma solicitação
exigir validação por especialista. Não alegue acessar sistemas, arquivos ou
fontes que não estejam presentes na conversa."""

CSS = """
:root {
  --aurora-green: #009c3b;
  --aurora-gold: #ffdf00;
  --aurora-blue: #002776;
  --aurora-ink: #07162b;
}
.gradio-container {
  background: radial-gradient(circle at top right, #123861, #07162b 48%, #03101e);
  color: #f8fbff;
  min-height: 100vh;
}
.aurora-header {
  border: 1px solid rgba(255, 223, 0, 0.45);
  border-radius: 18px;
  margin: 0.5rem 0 1.5rem;
  overflow: hidden;
  background: linear-gradient(115deg, rgba(0, 156, 59, 0.3), rgba(0, 39, 118, 0.7));
  box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
}
.aurora-header__content { padding: 2rem; }
.aurora-header h1 {
  color: var(--aurora-gold);
  font-size: 2.25rem;
  margin: 0;
}
.aurora-header p { margin: 0.65rem 0 0; font-size: 1.05rem; }
.aurora-note {
  border-left: 4px solid var(--aurora-gold);
  background: rgba(255, 255, 255, 0.07);
  border-radius: 8px;
  padding: 0.9rem 1rem;
}
footer { visibility: hidden; }
"""


def _configured_client() -> OpenAI:
    """Cria um cliente OpenAI usando exclusivamente o segredo do ambiente."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "A variável secreta OPENAI_API_KEY não está configurada neste Space."
        )
    return OpenAI(api_key=api_key)


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Converte o histórico do Gradio para o formato aceito pelo modelo."""
    messages: list[dict[str, str]] = []
    for item in history or []:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        messages.append({"role": role, "content": content})
    return messages


def respond(message: str, history: list[dict[str, Any]] | None) -> str:
    """Envia a mensagem ao modelo configurado e devolve a resposta textual."""
    clean_message = message.strip()
    if not clean_message:
        raise gr.Error("Escreva uma mensagem antes de enviar.")
    if len(clean_message) > MAX_MESSAGE_CHARS:
        raise gr.Error(
            f"A mensagem excede o limite de {MAX_MESSAGE_CHARS:,} caracteres."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": clean_message})

    try:
        response = _configured_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            messages=messages,
            temperature=0.35,
        )
    except RuntimeError as exc:
        raise gr.Error(str(exc)) from exc
    except RateLimitError as exc:
        raise gr.Error(
            "Limite temporário do provedor. Tente novamente em instantes."
        ) from exc
    except APIConnectionError as exc:
        raise gr.Error(
            "Não foi possível alcançar o provedor de IA."
        ) from exc
    except APIError as exc:
        raise gr.Error(
            "O provedor de IA retornou um erro ao processar a solicitação."
        ) from exc

    content = response.choices[0].message.content
    return content or "Não foi possível gerar uma resposta para esta solicitação."


with gr.Blocks(title=APP_TITLE, css=CSS, theme=gr.themes.Base()) as demo:
    gr.HTML(
        """
        <section class="aurora-header">
          <div class="aurora-header__content">
            <h1>Chat-GPT Aurora</h1>
            <p>Interface de IA em português para análises, pesquisa e produtividade.</p>
          </div>
        </section>
        """
    )
    gr.Markdown(
        """
        <div class="aurora-note">
        <strong>Uso responsável:</strong> não envie dados pessoais, sigilosos ou
        estratégicos. Esta demonstração utiliza a chave do provedor configurada
        como segredo no Space;
        portanto, ela não garante residência nacional de dados por si só.
        </div>
        """
    )
    gr.ChatInterface(
        fn=respond,
        type="messages",
        chatbot=gr.Chatbot(
            height=480,
            type="messages",
            placeholder="Como posso ajudar?",
        ),
        textbox=gr.Textbox(
            placeholder="Digite sua pergunta em português…",
            label="Mensagem",
            lines=2,
        ),
        examples=[
            "Explique, em linguagem simples, o que é soberania digital.",
            "Crie uma lista de verificação para uma auditoria de dados.",
            (
                "Resuma estes objetivos em um plano de ação: segurança, "
                "privacidade e transparência."
            ),
        ],
    )
    gr.Markdown(
        """
        **Transparência técnica.** O modelo padrão é configurável pela variável
        `OPENAI_MODEL`. Para operar com modelos ou infraestrutura própria, substitua
        o adaptador do provedor em `app.py` e mantenha as credenciais exclusivamente
        nos segredos da plataforma.
        """
    )


if __name__ == "__main__":
    demo.launch()

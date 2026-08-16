"""Componentes visuais do Chat-GPT Aurora."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from aurora_app.service import (
    AuroraConfigurationError,
    AuroraProviderError,
    complete_chat,
)

APP_TITLE = "Chat-GPT Aurora"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
BANNER_PATH = ASSETS_DIR / "banner_chatgpt.png"

CSS = """
:root {
  --aurora-green: #009c3b;
  --aurora-gold: #ffdf00;
  --aurora-blue: #002776;
  --aurora-ink: #061326;
  --aurora-surface: #0c213c;
  --aurora-text: #f5f8ff;
}
.gradio-container {
  background:
    radial-gradient(circle at 12% 0%, rgba(0, 156, 59, 0.28), transparent 31rem),
    radial-gradient(circle at 90% 9%, rgba(0, 39, 118, 0.65), transparent 32rem),
    var(--aurora-ink);
  color: var(--aurora-text);
  max-width: 1240px !important;
  padding-bottom: 3rem !important;
}
#aurora-shell {
  border: 1px solid rgba(255, 223, 0, 0.42);
  background: rgba(7, 20, 38, 0.85);
  border-radius: 24px;
  box-shadow: 0 25px 70px rgba(0, 0, 0, 0.35);
  overflow: hidden;
}
.aurora-hero { position: relative; background: #07162b; }
.aurora-hero img {
  display: block;
  height: 250px;
  object-fit: cover;
  opacity: 0.7;
  width: 100%;
}
.aurora-hero__overlay {
  background: linear-gradient(90deg, rgba(0, 39, 118, 0.93), rgba(0, 156, 59, 0.64));
  inset: 0;
  padding: 2rem;
  position: absolute;
}
.aurora-eyebrow {
  color: var(--aurora-gold);
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.aurora-hero h1 { color: #fff; font-size: 2.3rem; margin: 0.55rem 0; }
.aurora-hero p { max-width: 680px; margin: 0; font-size: 1.05rem; }
.aurora-badges { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 1rem; }
.aurora-badge {
  background: rgba(0, 0, 0, 0.32);
  border: 1px solid rgba(255, 223, 0, 0.55);
  border-radius: 999px;
  color: var(--aurora-gold);
  font-size: 0.76rem;
  font-weight: 750;
  padding: 0.35rem 0.7rem;
}
.aurora-content { padding: 1.25rem 1.5rem 1.75rem; }
.aurora-note {
  background: rgba(255, 223, 0, 0.08);
  border-left: 4px solid var(--aurora-gold);
  border-radius: 8px;
  color: #f4f7fc;
  margin-bottom: 1rem;
  padding: 0.8rem 1rem;
}
.aurora-footer {
  border-top: 1px solid rgba(255, 255, 255, 0.13);
  color: #bdc9dd;
  font-size: 0.85rem;
  margin-top: 1.5rem;
  padding-top: 1rem;
  text-align: center;
}
footer { display: none !important; }
"""


def _chat_response(
    message: str,
    history: list[dict[str, Any]] | None,
) -> str:
    """Adapta erros de domínio para mensagens compreensíveis na interface."""
    try:
        return complete_chat(message, history)
    except (ValueError, AuroraConfigurationError, AuroraProviderError) as exc:
        raise gr.Error(str(exc)) from exc


def _hero_markup() -> str:
    """Monta o cabeçalho visual, mantendo um fallback quando não há imagem."""
    banner = ""
    if BANNER_PATH.exists():
        banner = f'<img src="/gradio_api/file={BANNER_PATH}" alt="Chat-GPT Aurora">'

    return f"""
    <section class="aurora-hero">
      {banner}
      <div class="aurora-hero__overlay">
        <div class="aurora-eyebrow">Inteligência • Privacidade • Transparência</div>
        <h1>Chat-GPT Aurora</h1>
        <p>Uma interface de IA em português para pesquisa, análise e produtividade.</p>
        <div class="aurora-badges">
          <span class="aurora-badge">PORTUGUÊS DO BRASIL</span>
          <span class="aurora-badge">USO RESPONSÁVEL</span>
          <span class="aurora-badge">CREDENCIAIS PROTEGIDAS</span>
        </div>
      </div>
    </section>
    """


def build_demo() -> gr.Blocks:
    """Cria a interface Gradio pronta para execução local ou no Space."""
    with gr.Blocks(title=APP_TITLE, css=CSS) as demo:
        with gr.Column(elem_id="aurora-shell"):
            gr.HTML(_hero_markup())
            with gr.Column(elem_classes=["aurora-content"]):
                gr.HTML(
                    """
                    <div class="aurora-note">
                      <strong>Uso responsável:</strong> não envie credenciais, dados
                      pessoais sensíveis ou informações confidenciais. A residência e
                      a retenção dos dados dependem do provedor e da implantação.
                    </div>
                    """
                )
                gr.ChatInterface(
                    fn=_chat_response,
                    type="messages",
                    chatbot=gr.Chatbot(
                        height=500,
                        type="messages",
                        placeholder="Inicie uma conversa com a Aurora.",
                        allow_tags=False,
                    ),
                    textbox=gr.Textbox(
                        label="Mensagem",
                        placeholder="Escreva sua pergunta em português…",
                        lines=2,
                    ),
                    examples=[
                        "Explique soberania digital em linguagem simples.",
                        "Crie uma lista de verificação para auditoria de dados.",
                        (
                            "Transforme estes objetivos em um plano: "
                            "segurança, privacidade e transparência."
                        ),
                    ],
                )
                gr.HTML(
                    """
                    <div class="aurora-footer">
                      Chat-GPT Aurora · interface demonstrativa com modelo e
                      provedor configuráveis por variáveis de ambiente.
                    </div>
                    """
                )
    return demo

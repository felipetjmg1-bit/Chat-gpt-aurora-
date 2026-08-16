"""Ponto de entrada do Chat-GPT Aurora para Hugging Face Spaces."""

from aurora_app.ui import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()

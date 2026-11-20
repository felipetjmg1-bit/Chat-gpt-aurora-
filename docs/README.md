# Documentação técnica

## Fluxo principal do chat
`app.py` cria a interface Gradio; `aurora_app/ui.py` monta os componentes; `aurora_app/service.py` prepara o histórico e chama a API OpenAI; `aurora_app/config.py` lê `OPENAI_API_KEY` e `OPENAI_MODEL`.

## Trilha Speckle Automate
`main.py` e `flatten.py` representam uma integração distinta, com dependências e variáveis Speckle próprias. Não é etapa obrigatória para executar a interface Gradio.

## Limites e privacidade
As mensagens enviadas ao chat são transmitidas ao provedor OpenAI. O projeto não implementa autenticação multiusuário ou armazenamento próprio de conversas. Não use dados sensíveis sem base legal, transparência e avaliação contratual/técnica do provedor.

Antes de publicar, revise as configurações do Space, segredos, logs, limites de uso e políticas de retenção.

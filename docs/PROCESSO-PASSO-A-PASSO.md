# AURORA CHAT IA · fluxo passo a passo

Interface conversacional em português com provedor configurável.

## 1. CONFIGURAR ACESSO

Definir OPENAI_API_KEY e OPENAI_MODEL no ambiente.

## 2. INICIAR A INTERFACE

Executar app.py e carregar a interface Gradio.

## 3. ENVIAR MENSAGEM

A pessoa escreve uma pergunta na conversa.

## 4. PROCESSAR NO SERVIÇO

aurora_app/service.py encaminha a solicitação ao provedor.

## 5. EXIBIR RESPOSTA

A interface mostra o retorno e mantém histórico limitado.
## Resultado e limites

Experiência conversacional ligada a serviço externo. Requer chave e conectividade.

Este guia descreve o fluxo documentado no repositório. Recursos futuros devem ser identificados como planejados, e não como funcionalidades já disponíveis.

# Aurora Chat IA

<p align="center"><img src="PROJECT-OVERVIEW.svg" alt="Diagrama de funcionalidades de Aurora Chat IA" width="100%"></p>


**[Aplicação conversacional funcional]** Interface conversacional em português construída com Gradio e integrada à API OpenAI. A aplicação depende de uma chave válida do provedor e de conectividade externa. Não é um modelo local nem oferece soberania ou residência de dados por si só.

## Problema e objetivo

Disponibilizar uma interface simples para enviar mensagens e um histórico limitado a um modelo configurado por variável de ambiente. O projeto pode servir como demonstração de integração de aplicações de IA para usuários e empresas brasileiras.

## O que está implementado

- Interface Gradio montada em `aurora_app/ui.py`.
- Serviço de conversação em `aurora_app/service.py`.
- Configuração por `OPENAI_API_KEY` e `OPENAI_MODEL` em `aurora_app/config.py`.
- Ponto de entrada `app.py`.
- Testes existentes em `tests/`.
- Uma trilha Speckle Automate separada, com arquivos `main.py` e `flatten.py`; ela não faz parte do fluxo de chat Gradio.

## Como executar

Requisitos: Python 3.11 ou superior.

```bash
git clone https://github.com/felipetjmg1-bit/Chat-gpt-aurora-.git
cd Chat-gpt-aurora-
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Defina a chave no ambiente sem colocá-la no código ou no Git:

```bash
export OPENAI_API_KEY="SUA_CHAVE_LOCAL"
export OPENAI_MODEL="gpt-4o-mini"
python app.py
```

No Windows PowerShell, use `$env:OPENAI_API_KEY="SUA_CHAVE_LOCAL"` e `$env:OPENAI_MODEL="gpt-4o-mini"`. A interface será disponibilizada pelo Gradio no endereço local informado no terminal.

## Exemplo prático

Na interface, envie uma pergunta como:

```text
Explique soberania digital em linguagem simples.
```

A aplicação encaminha a mensagem e o histórico limitado ao serviço configurado na API OpenAI. A resposta depende da disponibilidade do provedor, da chave, do modelo selecionado e dos limites da conta. Este exemplo descreve o fluxo do código; não é uma saída de execução capturada.

## Estrutura

- `app.py`: inicialização da interface.
- `aurora_app/`: configuração, serviço e UI.
- `tests/`: testes automatizados existentes.
- `example.function_inputs.json`: exemplo de entrada de função.
- `main.py` e `flatten.py`: trilha separada de Speckle Automate.
- `.env.example`: nomes de variáveis sem credenciais válidas.
- `requirements.txt` e `pyproject.toml`: dependências de trilhas distintas; confira os respectivos escopos antes de instalar extras.

## Segurança, privacidade e limitações

- A chave da API deve ser mantida em segredo do ambiente ou no gerenciador de secrets do serviço de hospedagem. Nunca a publique em commits, issues ou capturas de tela.
- Mensagens e histórico são enviados ao provedor externo configurado. Avalie os termos, retenção, região, contrato e configurações de privacidade do provedor antes de tratar dados pessoais.
- Não envie dados pessoais sensíveis, credenciais, informações sigilosas ou dados de terceiros sem base legal, autorização e avaliação adequada.
- O repositório não implementa, por si só, autenticação corporativa, controle multiusuário, gestão de consentimento, política de retenção ou residência nacional de dados.
- O modelo padrão é configurável; a aplicação não comprova precisão, imparcialidade ou adequação para decisões de alto impacto.
- A trilha Speckle Automate é independente e exige configuração própria. Não confunda sua existência com funcionalidade integrada ao chat.

## Testes e dependências

Execute `pytest -q` para rodar os testes presentes. Não afirmamos que tenham sido executados nesta PR. `requirements.txt` declara faixas de Gradio e OpenAI SDK, não um lockfile com versões totalmente fixadas.

## Licença

Apache-2.0. Consulte [LICENSE](LICENSE).


**Empresa responsável:** Impulso Digital Soluções em IA.

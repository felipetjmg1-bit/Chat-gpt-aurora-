# Aurora Chat IA
**[PROTÓTIPO EXPERIMENTAL]** · Python · Gradio · OpenAI API · Apache-2.0

Interface conversacional em português construída com Gradio. O serviço envia a mensagem e um histórico limitado à API OpenAI, usando `gpt-4o-mini` como modelo padrão configurável. O repositório também contém uma trilha separada de função Speckle Automate para dados BIM, que não deve ser confundida com o fluxo principal do chat.

## Componentes presentes
- Interface Gradio: `app.py` e `aurora_app/ui.py`.
- Serviço de conversa: `aurora_app/service.py`.
- Configuração por ambiente: `aurora_app/config.py`.
- Limites de tamanho e histórico definidos no código.
- Código separado de automação Speckle em `main.py` e `flatten.py`.
- Testes em `tests/`.

## Executar em cinco minutos
Requer Python 3.11 ou superior.

```bash
git clone https://github.com/felipetjmg1-bit/Chat-gpt-aurora-.git
cd Chat-gpt-aurora-
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Configure a chave no ambiente local. Não cole a chave em arquivos versionados:
```bash
export OPENAI_API_KEY="sua-chave"
export OPENAI_MODEL="gpt-4o-mini"
python app.py
```
No Windows PowerShell, use `$env:OPENAI_API_KEY="sua-chave"` e `$env:OPENAI_MODEL="gpt-4o-mini"`. O arquivo `.env.example` também contém variáveis de configuração para a integração Speckle; não são necessárias para iniciar o chat Gradio.

## Exemplo prático
Na interface, envie uma pergunta como:
```text
Explique soberania digital em linguagem simples.
```
A aplicação encaminha a mensagem e o histórico limitado ao modelo configurado. A resposta depende da disponibilidade, credenciais, políticas e comportamento da API externa.

## Testes
```bash
pytest
```

## Segurança, privacidade e limitações
- A chave `OPENAI_API_KEY` é credencial sensível: mantenha-a em variável de ambiente ou gerenciador de segredos.
- As mensagens e o histórico enviados à API são processados pelo provedor externo; não prometa residência local ou soberania integral dos dados.
- O repositório não implementa autenticação de usuários, gestão de contas, controle de acesso multiusuário ou armazenamento próprio de conversas no fluxo principal.
- Não envie dados pessoais, sigilosos ou de terceiros sem base legal, transparência e avaliação do provedor.
- A trilha Speckle Automate é componente distinto e requer configuração própria.
- Para LGPD, informe os titulares sobre a transferência a terceiros, defina finalidade/base legal, minimize conteúdo e estabeleça retenção.

## Licença
Apache-2.0. Consulte `LICENSE`.

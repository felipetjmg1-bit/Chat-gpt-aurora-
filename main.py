"""
Chat-GPT Aurora - Speckle Automate Function
Integrando IA (Aurora) para análise inteligente de dados BIM no Speckle.
"""

import os
from pydantic import Field, SecretStr
from speckle_automate import (
    AutomateBase,
    AutomationContext,
    execute_automate_function,
)
from openai import OpenAI
from flatten import flatten_base

class FunctionInputs(AutomateBase):
    """Parâmetros de entrada para a função Aurora AI."""
    
    openai_api_key: SecretStr = Field(
        title="OpenAI API Key",
        description="Chave para acessar o modelo Aurora/GPT para análise."
    )
    analysis_prompt: str = Field(
        default="Analise os seguintes objetos BIM e identifique possíveis inconsistências ou otimizações.",
        title="Prompt de Análise",
        description="O que você quer que a Aurora analise nos dados?"
    )

def automate_function(
    automate_context: AutomationContext,
    function_inputs: FunctionInputs,
) -> None:
    """
    Função que recebe dados do Speckle e os envia para análise via IA Aurora.
    """
    # 1. Receber dados do Speckle
    version_root_object = automate_context.receive_version()
    flat_objects = list(flatten_base(version_root_object))
    
    # 2. Preparar sumário dos dados para a IA
    # (Limitando para não exceder tokens em modelos menores)
    object_types = {}
    for obj in flat_objects[:100]:
        t = obj.speckle_type
        object_types[t] = object_types.get(t, 0) + 1
    
    data_summary = f"Total de objetos analisados (amostra): {len(flat_objects[:100])}\n"
    data_summary += "Tipos encontrados:\n"
    for t, count in object_types.items():
        data_summary += f"- {t}: {count}\n"

    # 3. Chamar a API da OpenAI (Aurora)
    try:
        client = OpenAI(api_key=function_inputs.openai_api_key.get_secret_value())
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Usando um modelo eficiente
            messages=[
                {"role": "system", "content": "Você é a Aurora, uma especialista em análise de dados BIM e Speckle."},
                {"role": "user", "content": f"{function_inputs.analysis_prompt}\n\nDados do Modelo:\n{data_summary}"}
            ]
        )
        
        analysis_result = response.choices[0].message.content
        
        # 4. Anexar resultado ao Speckle
        automate_context.mark_run_success(f"Análise Aurora concluída: {analysis_result[:200]}...")
        
        # Salvar relatório completo como arquivo de resultado
        with open("relatorio_aurora.md", "w") as f:
            f.write(f"# Relatório de Análise Aurora AI\n\n{analysis_result}")
        
        automate_context.store_file_result("relatorio_aurora.md")

    except Exception as e:
        automate_context.mark_run_failed(f"Falha na integração com Aurora AI: {str(e)}")

if __name__ == "__main__":
    execute_automate_function(automate_function, FunctionInputs)

"""Chat-GPT Aurora - Speckle Automate Function.

Integra IA Aurora para análise inteligente de dados BIM no Speckle.
"""

from openai import OpenAI
from pydantic import Field, SecretStr
from speckle_automate import (
    AutomateBase,
    AutomationContext,
    execute_automate_function,
)

from flatten import flatten_base


class FunctionInputs(AutomateBase):
    """Parâmetros de entrada para a função Aurora AI."""

    openai_api_key: SecretStr = Field(
        title="OpenAI API Key",
        description="Chave para acessar o modelo Aurora/GPT para análise.",
    )
    analysis_prompt: str = Field(
        default=(
            "Realize uma auditoria técnica rigorosa. Verifique se há "
            "duplicidade de IDs, inconsistências de materiais e se a "
            "hierarquia espacial faz sentido para um modelo de construção."
        ),
        title="Prompt de Análise Avançada",
        description="Instruções específicas para a auditoria de IA.",
    )


def automate_function(
    automate_context: AutomationContext,
    function_inputs: FunctionInputs,
) -> None:
    """Recebe dados do Speckle e os envia para análise via IA Aurora."""
    version_root_object = automate_context.receive_version()
    flat_objects = list(flatten_base(version_root_object))

    object_types: dict[str, int] = {}
    missing_params: list[str] = []
    for obj in flat_objects[:150]:
        object_type = obj.speckle_type
        object_types[object_type] = object_types.get(object_type, 0) + 1

        if "Structure" in object_type and not hasattr(obj, "material"):
            missing_params.append(
                f"Objeto {obj.id} ({object_type}) sem material definido."
            )

    data_summary = "Relatório de Dados BIM:\n"
    data_summary += f"- Total de objetos: {len(flat_objects)}\n"
    data_summary += (
        f"- Amostra para análise profunda: {len(flat_objects[:150])}\n"
    )
    data_summary += "Distribuição de tipos:\n"
    for object_type, count in object_types.items():
        data_summary += f"  * {object_type}: {count}\n"

    if missing_params:
        data_summary += "\nInconsistências detectadas por regras locais:\n"
        data_summary += "\n".join(missing_params[:10])

    try:
        client = OpenAI(
            api_key=function_inputs.openai_api_key.get_secret_value()
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é a Aurora, uma especialista em análise de "
                        "dados BIM e Speckle."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{function_inputs.analysis_prompt}\n\n"
                        f"Dados do Modelo:\n{data_summary}"
                    ),
                },
            ],
        )
        analysis_result = response.choices[0].message.content or ""

        automate_context.mark_run_success(
            f"Análise Aurora concluída: {analysis_result[:200]}..."
        )

        with open("relatorio_aurora.md", "w", encoding="utf-8") as report_file:
            report_file.write(
                f"# Relatório de Análise Aurora AI\n\n{analysis_result}"
            )

        automate_context.store_file_result("relatorio_aurora.md")
    except Exception as exc:
        automate_context.mark_run_failed(
            f"Falha na integração com Aurora AI: {exc}"
        )


if __name__ == "__main__":
    execute_automate_function(automate_function, FunctionInputs)

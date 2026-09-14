import re
import json
from typing import Optional, List
from pydantic import BaseModel, ValidationError, Field

# ---- 1. Tools ----
def buscar_preco(produto: str) -> str:
    precos = {"notebook": 3500.00, "mouse": 80.00, "teclado": 250.00}
    preco = precos.get(produto.lower())
    return f"R$ {preco:.2f}" if preco else f"Produto '{produto}' não encontrado."

def converter_moeda(valor: float, moeda_origem: str, moeda_destino: str) -> str:
    taxas = {("BRL", "USD"): 0.19, ("USD", "BRL"): 5.25}
    taxa = taxas.get((moeda_origem.upper(), moeda_destino.upper()))
    return f"{valor * taxa:.2f} {moeda_destino}" if taxa else "Conversão não suportada."

TOOLS_DISPONIVEIS = {
    "buscar_preco": buscar_preco,
    "converter_moeda": converter_moeda,
}

# ---- 2. Schema Pydantic ----
class ToolCall(BaseModel):
    tool: str
    arguments: dict

class ReActResponse(BaseModel):
    thought: str
    action: str  # "tool" ou "final"
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None
    tarefas_pendentes: List[str] = Field(default_factory=list)

# ---- 3. Prompt ----
SYSTEM_PROMPT = """
Você é um agente que resolve perguntas usando ferramentas, seguindo o ciclo Thought -> Action -> Observation.

REGRAS DE FUNCIONAMENTO E DECISÃO:
1. No PRIMEIRO turno (sem nenhuma Observation no histórico), decomponha a pergunta do usuário
   em uma lista de sub-tarefas necessárias para respondê-la por completo, e coloque essa lista
   em "tarefas_pendentes". Se a pergunta exigir só uma coisa, a lista terá 1 item.
2. A CADA novo turno, releia "tarefas_pendentes" do turno anterior. Se a última Observation
   já resolveu uma dessas tarefas, REMOVA essa tarefa da lista e devolva a lista atualizada
   (menor) em "tarefas_pendentes".
3. Você só pode usar "action": "final" quando "tarefas_pendentes" estiver VAZIA ([]).
   Se ainda restar qualquer item na lista, use "action": "tool" para resolver o próximo item.
4. Quando "action" for "final", "final_answer" deve usar os valores reais obtidos nas
   Observations anteriores (nunca invente um valor).
5. NUNCA repita uma tool com os mesmos argumentos que já apareceram numa Observation anterior.

Ferramentas disponíveis:
- buscar_preco(produto: str)
- converter_moeda(valor: float, moeda_origem: str, moeda_destino: str)
  moeda_origem e moeda_destino devem ser exatamente "BRL" ou "USD".

Responda SEMPRE em JSON válido, neste formato:
{"thought": "...",
 "action": "tool"|"final",
 "tool_call": {"tool": "...", "arguments": {...}} ou null,
 "final_answer": "..." ou null,
 "tarefas_pendentes": ["tarefa que ainda falta", ...] ou []}

EXEMPLO (com uma ferramenta fictícia "buscar_dados", só para ilustrar o FORMATO — não é uma ferramenta real disponível para você):
[Pergunta]: "Qual a capital da França e a população de lá?"
[Turno 1]:
{"thought": "Preciso descobrir a capital e depois a população dela.",
 "action": "tool",
 "tool_call": {"tool": "buscar_dados", "arguments": {"consulta": "capital da França"}},
 "final_answer": null,
 "tarefas_pendentes": ["descobrir a capital", "descobrir a população da capital"]}
[Observation]: "A capital da França é Paris."
[Turno 2]:
{"thought": "Já sei a capital (Paris). Falta a população dela.",
 "action": "tool",
 "tool_call": {"tool": "buscar_dados", "arguments": {"consulta": "população de Paris"}},
 "final_answer": null,
 "tarefas_pendentes": ["descobrir a população da capital"]}
[Observation]: "A população de Paris é de 2,1 milhões de habitantes."
[Turno 3]:
{"thought": "Já tenho a capital e a população. Todas as tarefas foram concluídas.",
 "action": "final",
 "tool_call": null,
 "final_answer": "A capital da França é Paris e sua população é de 2,1 milhões de habitantes.",
 "tarefas_pendentes": []}
"""



def extrair_primeiro_json(texto: str) -> str:
    inicio = texto.find("{")
    if inicio == -1:
        raise ValueError("Nenhum '{' encontrado na resposta do modelo.")
    
    profundidade = 0
    for i in range(inicio, len(texto)):
        if texto[i] == "{":
            profundidade += 1
        elif texto[i] == "}":
            profundidade -= 1
            if profundidade == 0:
                return texto[inicio:i + 1]
    
    raise ValueError("JSON não fechado corretamente na resposta do modelo.")

def parse_resposta(texto_bruto: str) -> ReActResponse:
    json_str = extrair_primeiro_json(texto_bruto)
    dados = json.loads(json_str)
    return ReActResponse(**dados)



# ---- 5. Loop com guardrail ----
def rodar_agente(pergunta: str, llm_generate_fn, max_ciclos=5):
    historico = f"{SYSTEM_PROMPT}\n\nPergunta: {pergunta}\n"
    print(f"\n================ 🚀 NOVA PERGUNTA: {pergunta} ================")

    for ciclo in range(1, max_ciclos + 1):
        print(f"\n--- 🔄 Ciclo {ciclo}/{max_ciclos} ---")
        texto_bruto = llm_generate_fn(historico)

        try:
            resposta = parse_resposta(texto_bruto)
        except (ValueError, ValidationError) as e:
            print(f"⚠️ [Erro de formato/JSON]: {e}")
            historico += f"\n[Erro de formato: {e}. Responda em JSON válido, seguindo o schema exato.]\n"
            continue

        print(f"🧠 [Thought]: {resposta.thought}")
        print(f"📋 [Tarefas Pendentes]: {resposta.tarefas_pendentes}")

        # ---- GUARDRAIL: não confia cegamente no action do modelo ----
        if resposta.action == "final" and len(resposta.tarefas_pendentes) > 0:
            print(f"🛑 [Guardrail]: modelo disse 'final' mas ainda há tarefas pendentes {resposta.tarefas_pendentes}. Forçando continuação.")
            historico += (
                f"\nAssistant: {texto_bruto}\n"
                f"[Erro: você marcou action=final mas tarefas_pendentes ainda não está vazia "
                f"({resposta.tarefas_pendentes}). Resolva a próxima tarefa pendente usando action=tool.]\n"
            )
            continue

        if resposta.action == "final":
            print(f"✅ [Resposta Final Gerada]: {resposta.final_answer}")
            return resposta.final_answer

        if resposta.tool_call:
            print(f"🛠️  [Tool Solicitada]: '{resposta.tool_call.tool}'")
            print(f"📦 [Argumentos Enviados]: {resposta.tool_call.arguments}")

            tool_fn = TOOLS_DISPONIVEIS.get(resposta.tool_call.tool)
            if tool_fn is None:
                print(f"❌ [Erro]: A ferramenta '{resposta.tool_call.tool}' não existe.")
                historico += f"\n[Erro: ferramenta '{resposta.tool_call.tool}' não existe.]\n"
                continue

            resultado = tool_fn(**resposta.tool_call.arguments)
            print(f"👁️  [Observation da Tool]: {resultado}")

            historico += f"\nAssistant: {texto_bruto}\nObservation: {resultado}\n"

    print("⚠️ Max ciclos atingido sem resposta final.")
    return "Não foi possível concluir em max_ciclos."

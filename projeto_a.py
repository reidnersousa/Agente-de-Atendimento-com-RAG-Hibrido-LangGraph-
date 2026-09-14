from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from agente import llm
from roteador import classificar_ticket, rotear_ticket, tratar_duvida, tratar_elogio
from tools import rodar_agente


# ---- Estado unificado do Projeto A ----
class AtendimentoState(TypedDict):
    ticket: str
    categoria: Optional[str]
    resposta: Optional[str]
    contexto_usado: Optional[list]
    historico_conversa: Annotated[list, add_messages]  # memória via thread_id


# ---- Wrapper: bug agora usa o agente de Tools do Dia 2, não resposta fixa ----
def tratar_bug_com_tools(state: AtendimentoState):
    print("---Tratando BUG (via agente ReAct)---")

    def gerar_texto(prompt: str) -> str:
        return llm.generate(prompt, max_new_tokens=300, enable_thinking=False)["response"]

    resposta = rodar_agente(state["ticket"], gerar_texto, max_ciclos=5)
    return {"resposta": resposta}


# ---- Montagem do grafo ----
builder = StateGraph(AtendimentoState)

builder.add_node("classificar_ticket", classificar_ticket)
builder.add_node("tratar_bug", tratar_bug_com_tools)
builder.add_node("tratar_duvida", tratar_duvida)
builder.add_node("tratar_elogio", tratar_elogio)

builder.set_entry_point("classificar_ticket")

builder.add_conditional_edges(
    "classificar_ticket",
    rotear_ticket,
    {"bug": "tratar_bug", "duvida": "tratar_duvida", "elogio": "tratar_elogio"},
)

builder.add_edge("tratar_bug", END)
builder.add_edge("tratar_duvida", END)
builder.add_edge("tratar_elogio", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

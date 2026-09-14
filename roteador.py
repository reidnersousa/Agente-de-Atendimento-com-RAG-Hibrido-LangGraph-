from typing import TypedDict, Literal, Optional
import json

from agente import llm
from tools import extrair_primeiro_json
from rag import Searcher


class TicketState(TypedDict):
    ticket: str
    categoria: Optional[str]
    resposta: Optional[str]
    contexto_usado: Optional[list]


# ---- Parser da categoria (reaproveita o mesmo padrão robusto do Dia 2) ----
def parse_categoria(texto_bruto: str) -> str:
    dados = json.loads(extrair_primeiro_json(texto_bruto))
    categoria = dados["categoria"]
    if categoria not in ("bug", "duvida", "elogio"):
        raise ValueError(f"Categoria inesperada: {categoria}")
    return categoria


# ---- Nó: classificar ----
def classificar_ticket(state: TicketState):
    prompt = f"""
    Você é um classificador de tickets de suporte. Classifique o ticket em EXATAMENTE uma categoria:

    - bug: o usuário relata que algo no sistema/produto não está funcionando como deveria (erro, travamento, falha técnica).
    - duvida: o usuário faz uma pergunta sobre política, processo, procedimento ou "como fazer algo".
    - elogio: o usuário está dando um feedback positivo, sem pedir nada.

    Exemplos:
    Ticket: "O app fecha sozinho quando eu abro a câmera"
    {{"categoria": "bug"}}

    Ticket: "Recebi um e-mail suspeito pedindo minha senha, o que eu faço?"
    {{"categoria": "duvida"}}

    Ticket: "Qual o prazo para solicitar reembolso?"
    {{"categoria": "duvida"}}

    Ticket: "A nova versão ficou muito mais rápida, parabéns!"
    {{"categoria": "elogio"}}

    Responda apenas com o JSON, sem texto adicional.

    Ticket: {state['ticket']}
    """
    saida = llm.generate(prompt, enable_thinking=False, max_new_tokens=50)
    categoria = parse_categoria(saida["response"])
    print(f"🏷️  [Classificador] categoria: {categoria}")
    return {"categoria": categoria}


# ---- Roteamento condicional (nomes corretos, mapeados no add_conditional_edges) ----
def rotear_ticket(state: TicketState) -> Literal["bug", "duvida", "elogio"]:
    return state["categoria"]


# ---- Nó: dúvida (agora usando o rag.py com ChromaDB) ----
searcher = Searcher()  # carrega 1x, reaproveitado em todas as chamadas

def tratar_duvida(state: TicketState):
    print("---Tratando DÚVIDA---")
    resultado = searcher.buscar_com_confianca(state["ticket"])
    nivel = resultado["nivel_confianca"]

    if nivel == "baixo":
        return {
            "resposta": "Não encontrei informações suficientes sobre esse tema nas políticas disponíveis. Por favor, consulte o RH diretamente.",
            "contexto_usado": [],
        }

    contexto = "\n\n".join(
        f"[Fonte: {chunk['source']}]\n{chunk['text']}"
        for chunk in resultado["chunks"]
    )

    instrucao_extra = (
        "Se o contexto não for suficiente para responder com certeza, avise o usuário disso e sugira consultar o RH."
        if nivel == "medio"
        else "Responda de forma direta e objetiva."
    )

    prompt = f"""
    Responda a pergunta do usuário usando APENAS as informações do contexto abaixo. {instrucao_extra}
    Contexto: {contexto}
    Pergunta: {state['ticket']}"""

    saida = llm.generate(prompt, enable_thinking=False, max_new_tokens=200)
    return {
        "resposta": saida["response"],
        "contexto_usado": [c["text"] for c in resultado["chunks"]],
    }


# ---- Nó: elogio ----
def tratar_elogio(state: TicketState):
    print("---Tratando ELOGIO---")
    return {"resposta": "Muito obrigado pelo feedback positivo!"}


# ---- Nó: bug ----
def tratar_bug(state: TicketState):
    print("---Tratando BUG---")
    prompt = f"Escreva uma resposta curta e profissional para este ticket de bug:\n{state['ticket']}"
    saida = llm.generate(prompt, enable_thinking=False, max_new_tokens=150)
    return {"resposta": saida["response"]}

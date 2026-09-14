# Agente de Atendimento com RAG Híbrido (LangGraph)

Agente de atendimento ao cliente que classifica tickets, responde dúvidas sobre políticas internas usando RAG híbrido (busca vetorial + BM25 + reranking), executa ações via tool calling, e mantém memória de conversa — construído com LangGraph e um LLM local (Qwen3 1.7B).

> 🔗 **Rodado no Kaggle:** [Agente de Atendimento com RAG Híbrido (LangGraph)](https://www.kaggle.com/code/reidnersantos/agente-de-atendimento-com-rag-h-brido-langgraph)

---

## O que este projeto demonstra

- Orquestração de agentes com **LangGraph** (`StateGraph`, roteamento condicional, memória por `thread_id`)
- **RAG híbrido** sobre documentos PDF reais: busca vetorial (ChromaDB) + busca por palavra-chave (BM25) + fusão por Reciprocal Rank Fusion (RRF) + reranking com CrossEncoder
- Sistema de **confiança em 3 níveis** (alto/médio/baixo) com fallback anti-alucinação
- **Tool calling** no padrão ReAct (Thought → Action → Observation), com parsing de JSON robusto e guardrails contra o modelo pular etapas
- Uso de um **LLM local** (Qwen3, via KaggleHub) em vez de API paga — decisão documentada abaixo

---

## Arquitetura

```
Ticket do usuário
       │
       ▼
┌──────────────────┐
│ classificar_ticket│  ← LLM classifica em bug / duvida / elogio
└──────────────────┘
       │
   ┌───┴────┬─────────┐
   ▼        ▼         ▼
 [bug]   [duvida]  [elogio]
   │        │         │
   ▼        ▼         ▼
 Agente   RAG        Resposta
 ReAct    Híbrido    direta
 (tools)  (Chroma+
          BM25+
          rerank)
   │        │         │
   └───┬────┴─────────┘
       ▼
   Resposta final
   (persistida por thread_id via MemorySaver)
```

## Stack

| Camada | Tecnologia |
|---|---|
| Orquestração de agentes | LangGraph (`StateGraph`, `MemorySaver`, roteamento condicional) |
| LLM | Qwen3 1.7B, local, via KaggleHub |
| Banco vetorial | ChromaDB (embutido, persistido em disco) |
| Busca por palavra-chave | BM25 (`rank_bm25`) |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers) |
| Reranking | CrossEncoder `mmarco-mMiniLMv2-L12-H384-v1` |
| Fusão de rankings | Reciprocal Rank Fusion (RRF) |
| Extração de PDF | PyMuPDF (`fitz`) |

---

## Estrutura de arquivos

```
src/
├── agente.py              # carrega o Qwen3 local (KaggleHub)
├── indexar_politicas.py   # extrai PDFs, gera chunks, indexa no ChromaDB
├── rag.py                 # Searcher: busca híbrida + confiança em 3 níveis
├── tools.py                # agente ReAct com tool calling (ciclo Thought/Action/Observation)
├── roteador.py             # classificação de tickets + nós bug/duvida/elogio
└── projeto_a.py             # grafo completo: roteador + RAG + tools + memória
```

---

## Decisões de design

**LLM local em vez de API.** Optou-se por rodar Qwen3 1.7B localmente via KaggleHub em vez de usar Gemini/OpenAI. Isso elimina custo de API e dependência de internet para inferência, ao custo de respostas mais lentas e menos consistentes em formatação JSON — mitigado com parsing tolerante a erros (`extrair_primeiro_json`) e guardrails no loop do agente.

**RAG generativo em vez de matcher de FAQ.** A primeira abordagem cogitada foi um matcher de perguntas pré-escritas. Foi descartada em favor de RAG generativo real sobre os PDFs de política, para manter compatibilidade com avaliação de *faithfulness* (o quanto a resposta é fundamentada no contexto recuperado) — métrica central em frameworks como RAGAS.

**ChromaDB embutido em vez de solução cloud.** Roda inteiramente dentro do ambiente Kaggle, sem API key nem dependência de serviço externo, mantendo o índice vetorial persistido em disco.

---

## Achados técnicos

### 1. Rebalanceamento dos pesos de confiança híbrida

A fórmula de confiança combina o score de similaridade vetorial (`bi_score`) com o score do reranker (`rerank_score`):

```python
confianca = 0.6 * bi_score + 0.4 * rerank_score
```

A ponderação inicial (`0.3 × bi_score + 0.7 × rerank_score`) sub-representava buscas em que o reranker acertava a ordem mas com magnitude baixa, chegando a inverter a posição do resultado correto (5ª → 1ª colocação, antes do fix). Corrigido rebalanceando os pesos e reordenando pela confiança combinada em vez de só `rerank_score`. Validado com 6 queries de diagnóstico antes/depois.

### 2. Limitação de calibração em perguntas fora do domínio

Perguntas totalmente fora do escopo dos documentos indexados (ex: *"Converter 10 reais em dólares"*, quando os documentos são políticas de RH) podem receber confiança **"média"** em vez de **"baixa"**, caso compartilhem vocabulário superficial com os documentos (ex: menção a "reais"/R$). Isso permite que o modelo gere uma resposta parcialmente correta, mas com conteúdo não fundamentado no contexto — no caso testado, o modelo corretamente identificou não ter a informação, mas em seguida sugeriu sites externos (Banco Central, exchangerate.com) que não vieram do contexto recuperado.

**Causa raiz:** o retrieval (BM25 + embeddings) encontra similaridade lexical (palavras em comum), não necessariamente relevância semântica de domínio.

**Testado:** variação do threshold `min_score` entre 0.25 e 0.35 — o caso problemático (confiança 0.28) permanece "médio" na maior parte dessa faixa, mostrando que ajustar apenas o threshold não é uma solução robusta isolada.

**Mitigação não implementada ainda:** reforçar o prompt do nível "médio"/"baixo" para proibir explicitamente sugestões fora do contexto recuperado, em vez de apenas "avisar" sobre possível insuficiência de informação.

> Esse é exatamente o tipo de falha que métricas de **faithfulness** (RAGAS/DeepEval) são desenhadas para capturar formalmente — motivo pelo qual avaliação estruturada é o próximo passo natural deste projeto.

---

## Limitações conhecidas

- Classificador de tickets cobre apenas 3 categorias (`bug`, `duvida`, `elogio`) — perguntas que não se encaixam em nenhuma (ex: pedidos de ação/cálculo) são forçadas para a categoria mais próxima, podendo rotear incorretamente.
- Ver "Achados técnicos" acima para a limitação de calibração em perguntas fora do domínio.
- Memória de conversa (`MemorySaver`) é apenas em RAM — não persiste entre reinícios do kernel/processo.

## Como rodar

1. Abra o notebook no Kaggle (link acima) com **Internet: On**.
2. Rode as células de instalação de dependências.
3. Rode `indexar_politicas.py` uma vez para popular o ChromaDB a partir dos PDFs.
4. Rode `main_projeto_a.py` para ver o agente completo em ação.

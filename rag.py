import json
import logging
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class Searcher:
    """
    Busca híbrida (BM25 + ChromaDB + RRF + CrossEncoder rerank).
    Assume que indexar_politicas.py já populou o Chroma.
    """

    def __init__(
        self,
        persist_path: str = "/kaggle/working/chroma_politicas",
        collection_name: str = "politicas_internas",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        top_k: int = 5,
    ):
        self.top_k = top_k

        logging.info("Conectando ao ChromaDB...")
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_collection(collection_name)

        # BM25 ainda precisa dos textos em memória — puxa direto do Chroma
        logging.info("Reconstruindo índice BM25 a partir do Chroma...")
        tudo = self.collection.get(include=["documents", "metadatas"])
        self.chunks = [
            {"text": doc, **meta}
            for doc, meta in zip(tudo["documents"], tudo["metadatas"])
        ]
        self.bm25 = BM25Okapi([c["text"].lower().split() for c in self.chunks])

        self.reranker = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
        self.model = SentenceTransformer(model_name)
        logging.info("Searcher pronto.")

    def _busca_bm25(self, query: str, top_k: int = 20) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "bi_score": round(float(scores[i]), 4),
                "text": self.chunks[i]["text"],
                "source": self.chunks[i].get("source"),
                "page": self.chunks[i].get("page"),
                "section": self.chunks[i].get("section"),
            }
            for i in top_indices if scores[i] > 0
        ]

    def _busca_vetorial(self, query: str, top_k: int = 20) -> list[dict]:
        query_embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        resultado = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        candidatos = []
        for doc, meta, dist in zip(
            resultado["documents"][0], resultado["metadatas"][0], resultado["distances"][0]
        ):
            # Chroma com espaço "cosine" devolve distância; similaridade = 1 - distância
            candidatos.append({
                "bi_score": round(1 - dist, 4),
                "text": doc,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "section": meta.get("section"),
            })
        return candidatos

    def _rrf(self, *rankings: list[dict], k: int = 60) -> list[dict]:
        combined = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking):
                chave = f"{chunk['source']}::{chunk['section']}"
                if chave not in combined:
                    combined[chave] = {"chunk": chunk, "rrf_score": 0.0}
                combined[chave]["rrf_score"] += 1 / (k + rank + 1)
        sorted_items = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [item["chunk"] for item in sorted_items]

    def _sigmoid(self, x: float) -> float:
        return float(1 / (1 + np.exp(-x)))

    def _confianca_hibrida(self, bi_score: float, rerank_score: float) -> float:
        return round(0.6 * bi_score + 0.4 * rerank_score, 4)  # pesos calibrados no Dia 1

    def retrieve(self, query: str, top_k: int = 3, rerank: bool = True) -> dict:
        candidates_k = 20 if rerank else top_k
        vetorial = self._busca_vetorial(query, top_k=candidates_k)
        bm25 = self._busca_bm25(query, top_k=candidates_k)
        candidates = self._rrf(vetorial, bm25)

        if rerank:
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for candidate, rs in zip(candidates, rerank_scores):
                candidate["rerank_score"] = round(self._sigmoid(rs), 4)
                candidate["confianca"] = self._confianca_hibrida(candidate["bi_score"], candidate["rerank_score"])
            candidates.sort(key=lambda x: x["confianca"], reverse=True)

        return {
            "chunks": candidates[:top_k],
            "confianca": candidates[0].get("confianca", candidates[0]["bi_score"]),
        }

    def buscar_com_confianca(
        self, query: str, min_score: float = 0.35,
        low_confidence: float = 0.10
    ) -> dict:
        resultado = self.retrieve(query)
        confianca = resultado["confianca"]

        if confianca >= min_score:
            nivel = "alto"
        elif confianca >= low_confidence:
            nivel = "medio"
        else:
            nivel = "baixo"
            logging.warning(f"Confiança {confianca:.2f} muito baixa para: '{query}'")

        return {**resultado, "nivel_confianca": nivel}

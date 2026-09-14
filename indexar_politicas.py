
import os 
import re 
import json
import logging
import unicodedata
import fitz
import chromadb
from sentence_transformers import SentenceTransformer

### ---- funções de extração (iguais às que voce já tinha) ----

def clean_text(texto: str) -> str:
    texto = unicodedata.normalize("NFKC",texto)
    texto = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", "", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def get_politica(texto: str) -> str | None:
    padrao = r"(?i)política\s+(?:de|da|do)?\s*([\wÀ-ú]+)"
    resultado = re.search(padrao, texto)
    return resultado.group(1) if resultado else None


def get_section(texto:str) -> list[tuple[str,str]]:
    padrao = r"(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)"
    resultados = re.findall(padrao, texto, flags=re.DOTALL)
    resultados = [(num, conteudo.strip()) for num, conteudo in resultados]
    if not resultados:
        texto_limpo = texto.strip()
        return [("0", texto_limpo)] if texto_limpo else []
    return resultados

def load_pdfs(pdf_folder: str) -> list[dict]:
    chunks = []
    for documento in os.listdir(pdf_folder):
        if not documento.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_folder, documento)
        try:
            doc = fitz.open(path)
        except Exception as e:
            logging.error(f"Não foi possível abrir {documento}: {e}")
            continue

        for page_num in range(doc.page_count):
            texto = doc[page_num].get_text()
            if not texto.strip():
                continue
            politica = get_politica(texto)
            for num, conteudo in get_section(texto):
                texto_limpo = clean_text(conteudo)
                if len(texto_limpo) < 20:
                    continue
                chunks.append({
                    "source": documento,
                    "page": page_num,
                    "section": num,
                    "politica": politica or "geral",  # Chroma não aceita None em metadata
                    "text": texto_limpo,
                })
        doc.close()

    logging.info(f"Total de chunks gerados: {len(chunks)}")
    return chunks


def indexar_no_chroma(
    chunks: list[dict],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    persist_path: str = "/kaggle/working/chroma_politicas",
    collection_name: str = "politicas_internas",
):
    """
    Gera embeddings e indexa direto no ChromaDB (substitui o np.save antigo).
    """
    if not chunks:
        raise ValueError("Chunks vazios. Rode load_pdfs primeiro.")

    model = SentenceTransformer(model_name)

    # mesmo prefixo de contexto que você já usava
    textos_para_embedding = [
        f"Política: {c['source'].replace('.pdf', '')} | {c['text']}"
        for c in chunks
    ]
    embeddings = model.encode(
        textos_para_embedding, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    client = chromadb.PersistentClient(path=persist_path)
    # apaga coleção antiga se já existir, pra reindexar limpo
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],  # texto original, sem o prefixo
        metadatas=[
            {"source": c["source"], "page": c["page"], "section": c["section"], "politica": c["politica"]}
            for c in chunks
        ],
    )

    logging.info(f"{len(chunks)} chunks indexados no Chroma em {persist_path}")
    return collection


root = "/kaggle/input/datasets/reidnersantos/polticas-de-servio"
chunks = load_pdfs(root)
indexar_no_chroma(chunks)

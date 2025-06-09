import ir_datasets
import os
import pickle

DATASET_NAME = "beir/cqadupstack/gaming"
CACHE_FILE = "src/beir_gaming_cached.pkl"
CACHE_FILE_QUERIES_QRELS = "src/beir_queries_qrels_cached.pkl"

def load_beir_documents(limit=None):
    """
    Carga documentos del dataset 'beir/cqadupstack/gaming' usando caché local.

    Parámetros:
        limit (int | None): Número máximo de documentos a retornar. Si es None, retorna todos.

    Retorna:
        tuple:
            - list[str]: Lista de textos de documentos.
            - list[str]: Lista de IDs de documentos.
    """
    doc_texts = None
    doc_ids = None

    # Verificar si existe la versión cacheada
    if os.path.exists(CACHE_FILE):
        print("Loading dataset from cache...")
        with open(CACHE_FILE, "rb") as f:
            doc_texts, doc_ids = pickle.load(f)
    
    # Si no hay caché, cargar desde ir_datasets y guardar en caché
    if doc_texts is None or doc_ids is None:
        print("Loading dataset from ir_datasets and caching...")
        dataset = ir_datasets.load(DATASET_NAME)
        doc_texts = []
        doc_ids = []
        for doc in dataset.docs_iter():
            doc_ids.append(doc.doc_id)
            doc_texts.append(doc.text)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((doc_texts, doc_ids), f)

    # Aplicar límite si es necesario
    if limit is not None:
        return doc_texts[:limit], doc_ids[:limit]
    return doc_texts, doc_ids

def load_beir_queries_and_qrels(limit=None):
    """
    Carga queries y qrels del dataset 'beir/cqadupstack/gaming' usando caché local.

    Parámetros:
        limit (int | None): Número máximo de queries a retornar. Si es None, retorna todas.

    Retorna:
        tuple:
            - dict[str, str]: Diccionario de queries en formato {query_id: texto}.
            - dict[str, list[str]]: Diccionario de qrels en formato {query_id: [doc_id, ...]}.
    """
    # Verificar si existe la versión cacheada
    if os.path.exists(CACHE_FILE_QUERIES_QRELS):
        print("Loading queries and qrels from cache...")
        with open(CACHE_FILE_QUERIES_QRELS, "rb") as f:
            queries, qrels = pickle.load(f)
    else:
        print("Loading queries and qrels from ir_datasets and caching...")
        dataset = ir_datasets.load(DATASET_NAME)

        # Cargar queries
        queries = {}
        for query in dataset.queries_iter():
            queries[query.query_id] = query.text
        # Cargar qrels solo si la relevancia es mayor a 0
        qrels = {}
        for qrel in dataset.qrels_iter():
            if qrel.relevance > 0:
                qrels.setdefault(qrel.query_id, []).append(qrel.doc_id)

        # Guardar en caché
        with open(CACHE_FILE_QUERIES_QRELS, "wb") as f:
            pickle.dump((queries, qrels), f)
            
    # Aplicar límite si es necesario
    if limit is not None:
        limited_queries = dict(list(queries.items())[:limit])
        limited_qrels = {qid: qrels[qid] for qid in limited_queries if qid in qrels}
        return limited_queries, limited_qrels

    return queries, qrels
from src.dataset_loader import (
    load_beir_documents, 
    load_beir_queries_and_qrels)
from src.search_engine import (
    build_tf_idf_matrix,
    query_vectorizer,
    compute_cosine_similarity,
    build_bm25_model,
    compute_bm25_scores
)
from src.preprocessing import preprocess_documents
from src.perf_metrics import precision_recall_at_k
import os

# ───── Configuración ─────
TOP_K = 1
USE_BM25 = False

# ───── Carga y preprocesamiento ─────
print("Cargando documentos...")
documents, document_ids = load_beir_documents(limit=20000) 
queries, qrels = load_beir_queries_and_qrels(limit=10)

df = preprocess_documents(documents)
preprocessed_docs = df['prep_doc'].tolist()

# ───── Crear índices ─────
print("Construyendo índice TF-IDF...")
tfidf_matrix, tfidf_vectorizer = build_tf_idf_matrix(preprocessed_docs)

# ───── Construye modelo BM25 ─────
print("Construyedo modelo BM25..")
preprocessed_token_docs = preprocess_documents(documents, return_type='tokens')
bm25_model = build_bm25_model(preprocessed_token_docs)


# ───── Interfaz de consola ─────
while True:
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpia pantalla
    print("===== Sistema de Recuperación de Información =====")
    print("Seleccione el algoritmo de recuperación:")
    print("1. Similitud Coseno con TF-IDF")
    print("2. BM25")
    print("3. Evaluar automáticamente (TF-IDF)")
    print("4. Salir")




    choice = input("Opción: ").strip()
    
    if choice == '1':
        USE_BM25 = False
    elif choice == '2':
        USE_BM25 = True
    elif choice == '3':
        # ───── Evaluación automática ─────
        print("Cargando consultas y relevancias...")
        

        total_precision = 0.0
        total_recall = 0.0
        num_queries = 0

        for query_id, query_text in queries.items():
            # Preprocesar la consulta
            query_tokens = preprocess_documents([query_text])['prep_doc'].iloc[0]
            # Vectorizar la consulta
            query_vec = query_vectorizer(query_tokens, tfidf_vectorizer)
            # Realizar la búsqueda
            results, _ = compute_cosine_similarity(tfidf_matrix, query_vec, df['document'])
            # Tomar los TOP_K resultados
            top_results = results.head(TOP_K)
            retrieved_doc_ids = [document_ids[i] for i in top_results.index]

            # Extraer los documentos relevantes, considerando los posibles formatos de qrels
            qrel_item = qrels.get(query_id, [])
            if isinstance(qrel_item, dict):
                relevant_doc_ids = list(qrel_item.keys())
            else:
                relevant_doc_ids = qrel_item

            # Calcular precisión y recall para la consulta
            precision, recall = precision_recall_at_k(relevant_doc_ids, retrieved_doc_ids, TOP_K)
            total_precision += precision
            total_recall += recall
            num_queries += 1

        mean_precision = total_precision / num_queries if num_queries > 0 else 0
        mean_recall = total_recall / num_queries if num_queries > 0 else 0

        print("\n--- Resultado de la Evaluación ---")
        print(f"Consultas evaluadas: {num_queries}")
        print(f"Precisión promedio @ {TOP_K}: {mean_precision:.4f}")
        print(f"Recall promedio @ {TOP_K}: {mean_recall:.4f}")
        input("Presione Enter para continuar...")
        continue  # Regresa al menú principal


    elif choice == '4':
        print("Saliendo...")
        break
    else:
        print("Opción inválida.")
        input("Presione Enter para continuar...")
        continue
    

    # Submenú de consultas
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"=== Consulta ({'BM25' if USE_BM25 else 'TF-IDF'}) ===")
        print("Escribe tu consulta o escribe 'volver' para regresar al menú.")
        query = input("> ").strip()

        if query.lower() == 'volver':
            break
        elif len(query) < 2:
            print("Consulta muy corta.")
            input("Presione Enter para continuar...")
            continue
        
        # Procesamiento y búsqueda
        query_tokens = preprocess_documents([query])['prep_doc'].iloc[0]
        query_tokens_bm25 = preprocess_documents([query], return_type='tokens')[0]

        if USE_BM25:
            results = compute_bm25_scores(bm25_model, query_tokens_bm25, df["document"])
        if not USE_BM25:
            query_vec = query_vectorizer(query_tokens, tfidf_vectorizer)
            results, measured_time = compute_cosine_similarity(tfidf_matrix, query_vec, df['document'])


        print(f"\nResultados para: \"{query}\"\n")
        print(f"Su consulta se resolvió en {measured_time:.2f} segundos.\n")
        for i, row in results.head(TOP_K).iterrows():
            print(f"{i+1}. Score: {row['Similarity']:.4f}")
            print(f"   {row['Document'][:200]}...\n")

        input("Presione Enter para hacer otra consulta...")


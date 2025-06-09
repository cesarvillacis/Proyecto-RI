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
from src.preprocessing import preprocess_documents, preprocess_both
from src.perf_metrics import precision_recall_at_k, average_precision
import os
import time

# ───── Configuración ─────
# Número de documentos relevantes a recuperar por consulta
TOP_K = 5
# Algoritmo por defecto: False = TF-IDF, True = BM25
USE_BM25 = False

# ───── Carga y preprocesamiento ─────
documents, document_ids = load_beir_documents(limit=1000) 
queries, qrels = load_beir_queries_and_qrels(limit=50)

# Preprocesamiento textual de los documentos
df = preprocess_documents(documents)
preprocessed_docs = df['prep_doc'].tolist()
preprocessed_token_docs = preprocess_documents(documents, return_type='tokens')
preprocessed_queries = {qid: preprocess_both(qtext) for qid, qtext in queries.items()}

# ───── Crear índices ─────
print("Construyendo índice TF-IDF...")
tfidf_matrix, tfidf_vectorizer = build_tf_idf_matrix(preprocessed_docs)

# ───── Construye modelo BM25 ─────
print("Construyedo modelo BM25...")
bm25_model = build_bm25_model(preprocessed_token_docs)


# ───── Interfaz de consola ─────
while True:
    os.system('cls' if os.name == 'nt' else 'clear')  # Limpia pantalla
    print("===== Sistema de Recuperación de Información =====")
    print("Seleccione el algoritmo de recuperación:")
    print("1. Similitud Coseno con TF-IDF")
    print("2. BM25")
    print("3. Evaluar automáticamente (TF-IDF y BM25)")
    print("4. Salir")

    choice = input("Opción: ").strip()
    
    if choice == '1':
        USE_BM25 = False
    elif choice == '2':
        USE_BM25 = True
    elif choice == '3': 
        # ───── Evaluación automática ─────
        print("Ejecutando evaluación automática...")

        # Inicialización de variables acumuladoras para métricas
        total_precision_tfidf = 0.0
        total_recall_tfidf = 0.0
        total_precision_bm25 = 0.0
        total_recall_bm25 = 0.0
        num_queries = 0
        total_map_tfidf = 0.0
        total_map_bm25 = 0.0
        num_queries = 0

        start = time.time()
        for query_id, query_text in queries.items():
            # Obtener la versión preprocesada de la consulta
            query_clean, query_tokens = preprocessed_queries[query_id]

            # === Evaluación TF-IDF ===
            query_vec = query_vectorizer(query_clean, tfidf_vectorizer)
            results_tfidf, time_tfid = compute_cosine_similarity(tfidf_matrix, query_vec, df['document'])
            top_results_tfidf = results_tfidf.head(TOP_K)
            retrieved_tfidf_ids = [document_ids[i] for i in top_results_tfidf.index]

            # === Evaluación BM25 ===
            results_bm25, time_bm25 = compute_bm25_scores(bm25_model, query_tokens, df['document'])
            top_results_bm25 = results_bm25.head(TOP_K)
            retrieved_bm25_ids = [document_ids[i] for i in top_results_bm25.index]

            # === Cálculo de métricas ===
            # Obtener los documentos relevantes para la consulta
            relevant_doc_ids = qrels[query_id]

            # Calcular Precisión y Recall para cada modelo
            prec_tfidf, rec_tfidf = precision_recall_at_k(relevant_doc_ids, retrieved_tfidf_ids, TOP_K)
            prec_bm25, rec_bm25 = precision_recall_at_k(relevant_doc_ids, retrieved_bm25_ids, TOP_K)

            # Calcular Promedio de Precisión (MAP)
            ap_tfidf = average_precision(relevant_doc_ids, retrieved_tfidf_ids)
            ap_bm25 = average_precision(relevant_doc_ids, retrieved_bm25_ids)

            # Acumular métricas
            total_precision_tfidf += prec_tfidf
            total_recall_tfidf += rec_tfidf
            total_precision_bm25 += prec_bm25
            total_recall_bm25 += rec_bm25
            total_map_tfidf += ap_tfidf
            total_map_bm25 += ap_bm25

            num_queries += 1

        # === Promedios finales ===
        # Calcular métricas promedio sobre todas las consultas evaluadas
        mean_prec_tfidf = total_precision_tfidf / num_queries
        mean_rec_tfidf = total_recall_tfidf / num_queries
        mean_prec_bm25 = total_precision_bm25 / num_queries
        mean_rec_bm25 = total_recall_bm25 / num_queries
        mean_map_tfidf = total_map_tfidf / num_queries
        mean_map_bm25 = total_map_bm25 / num_queries

        end = time.time()

        # Mostrar resultados por consola
        print("\n--- Resultado de la Evaluación Automática ---")
        print(f"Consultas evaluadas: {num_queries}")
        print(f"\n[TF-IDF]")
        print(f"Precisión promedio @ {TOP_K}: {mean_prec_tfidf:.2f}")
        print(f"Recall promedio @ {TOP_K}:    {mean_rec_tfidf:.2f}")
        print(f"MAP:                         {mean_map_tfidf:.4f}")
        print(f"\n[BM25]")
        print(f"Precisión promedio @ {TOP_K}: {mean_prec_bm25:.2f}")
        print(f"Recall promedio @ {TOP_K}:    {mean_rec_bm25:.2f}")
        print(f"MAP:                         {mean_map_bm25:.4f}")
        print("tiempo promedio TF-IDF: {:.2f} segundos".format(time_tfid))
        print("tiempo promedio BM25: {:.2f} segundos".format(time_bm25))
        print(f"Tiempo total de evaluación: {end - start:.2f} segundos")
        input("\nPresione Enter para continuar...")
        continue

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
        query_clean, query_tokens = preprocess_both(query)

        if USE_BM25:
            results, measured_time = compute_bm25_scores(bm25_model, query_tokens, df["document"])
        else:
            query_vec = query_vectorizer(query_clean, tfidf_vectorizer)
            results, measured_time = compute_cosine_similarity(tfidf_matrix, query_vec, df['document'])

        print(f"\nResultados para: \"{query}\"\n")
        print(f"Su consulta se resolvió en {measured_time:.2f} segundos.\n")
        for i, row in results.head(TOP_K).iterrows():
            print(f"{i+1}. Score: {row['Similarity']:.4f}")
            print(f"   {row['Document'][:200]}...\n")

        input("Presione Enter para hacer otra consulta...")


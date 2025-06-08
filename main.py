from src.dataset_loader import load_beir_documents
from src.preprocessing import preprocess_documents
from src.search_engine import (
    build_tf_idf_matrix,
    query_vectorizer,
    compute_cosine_similarity,
    build_bm25_model,
    compute_bm25_scores
)
import os


# ───── Configuración ─────
TOP_K = 5
USE_BM25 = False

# ───── Carga y preprocesamiento ─────
print("Cargando documentos...")
documents = load_beir_documents(limit=1000) # reduce para pruebas más rápidas
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
    print("3. Salir")



    choice = input("Opción: ").strip()
    
    if choice == '1':
        USE_BM25 = False
    elif choice == '2':
        USE_BM25 = True
    elif choice == '3':
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
            results = compute_cosine_similarity(tfidf_matrix, query_vec, df['document'])


        print(f"\n🔍 Resultados para: \"{query}\"\n")
        for i, row in results.head(TOP_K).iterrows():
            print(f"{i+1}. Score: {row['Similarity']:.4f}")
            print(f"   {row['Document'][:200]}...\n")

        input("Presione Enter para hacer otra consulta...")


#print(tfidf_matrix)


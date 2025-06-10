from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.perf_metrics import execute_time
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi


def build_tf_matrix(data):
    """
    Construye la matriz de Frecuencia de Términos (TF) a partir de un conjunto de documentos.

    Parámetros:
        data (list[str] | pd.DataFrame): Lista de documentos o DataFrame con una columna de texto.

    Retorna:
        tuple:
            - pd.DataFrame: Matriz TF con frecuencia de cada término por documento.
            - CountVectorizer: Vectorizador utilizado para construir la matriz.
    """
    # Verificar si data es un DataFrame y convertirlo a lista si es necesario
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].tolist()

    # Construir la matriz TF
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(data)

    # Se obtienen los términos y se crea el DataFrame de salida
    terms = vectorizer.get_feature_names_out()
    tf_df = pd.DataFrame(X_counts.toarray(), columns=terms)

    return tf_df, vectorizer

def build_tf_idf_matrix(data):
    """
    Construye la matriz TF-IDF (Term Frequency - Inverse Document Frequency).

    Parámetros:
        data (list[str] | pd.DataFrame): Lista de documentos o DataFrame con una columna de texto.

    Retorna:
        tuple:
            - sparse matrix: Matriz TF-IDF en formato disperso.
            - TfidfVectorizer: Vectorizador utilizado.
    """
    # Si se recibe un DataFrame, se convierte en lista de texto
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].tolist()

    # Construir la matriz TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(data)

    # No convertir a DataFrame denso
    return X_tfidf, tfidf_vectorizer

def build_inverted_index(tfidf_matrix, tfidf_vectorizer):
    """
    Construye un índice invertido a partir de la matriz TF-IDF y el vectorizador.

    Parámetros:
        tfidf_matrix (sparse matrix): Matriz TF-IDF.
        tfidf_vectorizer (TfidfVectorizer): Vectorizador entrenado.

    Retorna:
        dict: Índice invertido {término: [índices de documentos]}
    """
    terms = tfidf_vectorizer.get_feature_names_out()
    inverted_index = {}
    matrix = tfidf_matrix.tocsc()  # Para acceso eficiente por columna

    for term_idx, term in enumerate(terms):
        doc_indices = matrix[:, term_idx].nonzero()[0]
        inverted_index[term] = doc_indices.tolist()
    return inverted_index

def query_vectorizer(query, vectorizer):
    """
    Vectoriza una consulta usando un vectorizador previamente entrenado.

    Parámetros:
        query (str): Consulta de entrada.
        vectorizer (TfidfVectorizer | CountVectorizer): Vectorizador previamente entrenado.

    Retorna:
        sparse matrix: Vector transformado de la consulta.
    """
    return vectorizer.transform([query])

@execute_time
def compute_cosine_similarity(matrix, query_vector, documents):
    """
    Calcula la similitud coseno entre una consulta vectorizada y una matriz de documentos.

    Parámetros:
        matrix (sparse matrix): Matriz TF-IDF o TF.
        query_vector (sparse matrix): Vector de la consulta.
        documents (list[str]): Lista de documentos originales.

    Retorna:
        pd.DataFrame: Resultados ordenados por similitud, incluyendo los documentos y sus puntajes.
    """
    similarities = cosine_similarity(matrix, query_vector).flatten()

    # Crear DataFrame con índices para mantener referencia al documento original
    results_df = pd.DataFrame({
        "Index": range(len(similarities)),
        "Similarity": similarities
    })

    # Ordener los resultados por similitud
    results_df = results_df.sort_values(by="Similarity", ascending=False)

    # Agregamos la columna 'Document' accediendo a su índice en la lista original
    results_df["Document"] = results_df["Index"].map(lambda i: documents[i])

    return results_df

def build_bm25_model(documents):
    """
    Construye un modelo BM25 a partir de una lista de documentos tokenizados.

    Parámetros:
        documents (list[list[str]]): Lista de documentos tokenizados (cada documento es una lista de tokens).

    Retorna:
        BM25Okapi: Modelo BM25 entrenado.
    """
    # Se crea el modelo BM25 utilizando el corpus preprocesado
    bm25 = BM25Okapi(documents)
    
    #retorna una intancia del modelo BM25 para luego poder extraer el score en funcion de una query
    return bm25

@execute_time
def compute_bm25_scores(bm25_model, query_tokens, documents):
    """
    Calcula los puntajes BM25 para una consulta y devuelve los resultados ordenados.

    Parámetros:
        bm25_model (BM25Okapi): Modelo BM25 previamente entrenado.
        query_tokens (list[str]): Consulta tokenizada.
        documents (list[str]): Lista de documentos originales.

    Retorna:
        pd.DataFrame: Resultados con columnas 'Document' y 'Similarity', ordenados de mayor a menor.
    """
    scores = bm25_model.get_scores(query_tokens)

    results_df = pd.DataFrame({
        "Document": documents,
        "Similarity": scores
    }).sort_values(by="Similarity", ascending=False)

    return results_df
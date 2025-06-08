from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import regexp_tokenize

def build_tf_matrix(data):
    """
    Función que retorna el dataframe de la matriz TF
    Recibe como parámetro una lista de documentos o un DataFrame con una columna de texto.
    Retorna un DataFrame con la matriz TF y el vectorizador utilizado.
    """
    # Verificar si data es un DataFrame y convertirlo a lista si es necesario
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].tolist()

    # Construir la matriz TF
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(data)

    # Obtener términos y construir el DataFrame de vectorizer (matríz TF)
    terms = vectorizer.get_feature_names_out()
    tf_df = pd.DataFrame(X_counts.toarray(), columns=terms)

    return tf_df, vectorizer

def build_tf_idf_matrix(data):
    """
    Función que retorna el dataframe de la matriz TF-IDF
    Recibe como parámetro una lista de documentos o un DataFrame con una columna de texto.
    Retorna un DataFrame con la matriz TF-IDF y el vectorizador utilizado.
    """
    # Verificar si data es un DataFrame y convertirlo a lista si es necesario
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].tolist()

    # Construir la matriz TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(data)

    # Obtener términos y construir el DataFrame de vectorizer (matríz TF-IDF)
    terms_tfidf = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=terms_tfidf)
    
    return tfidf_df, tfidf_vectorizer

def query_vectorizer(query, vectorizer):
    """
    Función que vectoriza una consulta 
    Recibe como parámetro una consulta y un vectorizador.
    Retorna un vector de consulta transformado.
    """
    #Vectoriza una consulta utilizando el vectorizador proporcionado
    return vectorizer.transform([query])

def compute_cosine_similarity(matrix, query_vector, documents):
    """
    Función que calcula la similitud coseno entre una matriz y un vector de consulta
    Recibe como parámetros una matriz (dataframe), un vector de consulta y una lista de documentos.
    Retorna un DataFrame con los documentos y sus similitudes ordenados de manera descendente.
    """
    # Calcula la similitud coseno entre la matriz y el vector de consulta
    similarities = cosine_similarity(matrix, query_vector).flatten()

    # Construcción de dataframe de resultados de manera descendente
    similar_documents = pd.DataFrame({
        "Document": documents,
        "Similarity": similarities})
    similar_documents = similar_documents.sort_values(by="Similarity", ascending=False)

    return similar_documents

def build_bm25_model(documents):
    """
    Construye el modelo BM25 a partir de una lista de documentos previamente tokenizados.
    
    Argumentos:
        documents (List[str]): Lista de documentos toeknizados, donde cada documento es una lista de tokens.
        
    Retorno:
        BM25Okapi: El modelo BM25 entrenado con los documentos proporcionados.
    """
    # Se crea el modelo BM25 utilizando el corpus preprocesado
    bm25 = BM25Okapi(documents)
    
    #retorna una intancia del modelo BM25 para luego poder extraer el score en funcion de una query
    return bm25

def compute_bm25_scores(bm25_model, query_tokens, documents):
    """
    Calcula los puntajes BM25 para una consulta y devuelve un DataFrame con los resultados ordenados.

    Args:
        bm25_model (BM25Okapi): Modelo BM25 ya entrenado.
        query_tokens (List[str]): Consulta preprocesada y tokenizada.
        documents (List[str]): Lista de documentos originales (texto plano).

    Returns:
        pd.DataFrame: Resultados con columnas 'Document' y 'Score', ordenados de mayor a menor relevancia.
    """
    scores = bm25_model.get_scores(query_tokens)

    results_df = pd.DataFrame({
        "Document": documents,
        "Similarity": scores
    }).sort_values(by="Similarity", ascending=False)

    return results_df




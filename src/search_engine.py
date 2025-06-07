from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Función que retorna el dataframe de la matriz TF
def build_tf_matrix(data):
    # Verificar si data es un DataFrame y convertirlo a lista si es necesario
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].tolist()

    # Construir la matriz TF
    vectorizer = CountVectorizer()
    X_counts = vectorizer.fit_transform(data)

    # Obtener términos y construir el DataFrame de vectorizer (matríz TF)
    terms = vectorizer.get_feature_names_out()
    tf_df = pd.DataFrame(X_counts.toarray(), columns=terms)

    return tf_df

# Función que retorna el dataframe de la matriz TF-IDF
def build_tf_idf_matrix(data):
    # Verificar si data es un DataFrame y convertirlo a lista si es necesario
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].tolist()

    # Construir la matriz TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(data)

    # Obtener términos y construir el DataFrame de vectorizer (matríz TF-IDF)
    terms_tfidf = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=terms_tfidf)
    
    return tfidf_df

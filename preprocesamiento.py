import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def remove_stopwords(tokens):
    sw = set(stopwords.words('english'))
    return [t for t in tokens if t not in sw]

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]

def preprocesamiento(docs):
    # Crear DataFrame
    df = pd.DataFrame(docs, columns=['document'])

    # Tokenización usando expresión regular
    df['regex_tokens'] = df['document'].str.lower().apply(
        lambda x: regexp_tokenize(x, pattern=r'\w[a-z]+')
    )

    # Eliminar stopwords
    df['sw_tokens'] = df['regex_tokens'].apply(remove_stopwords)

    # Lematización
    df['lem_tokens'] = df['sw_tokens'].apply(lemmatize)

    # Unir tokens en un solo string para la columna final
    df['prep_doc'] = df['lem_tokens'].str.join(' ')

    # Devolver solo columnas requeridas
    return df[['document', 'prep_doc']]

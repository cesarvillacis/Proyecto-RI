# archivo: preprocesamiento.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

print("Cargando recursos de NLTK...")

nltk.download('punkt')
nltk.download('punkt_tab')

# Descargar recursos necesarios (solo la primera vez)
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)

def preprocesamiento(docs):
    # ── Tokenización ──
    tokenized_docs = [word_tokenize(doc) for doc in docs]

    # ── Normalización ──
    normalized_docs = [
        [token.lower() for token in doc if re.fullmatch(r'[a-zA-Z]+', token)]
        for doc in tokenized_docs
    ]

    # ── Stopwords ──
    stop_words = set(stopwords.words('english'))
    filtered_docs = [
        [token for token in doc if token not in stop_words]
        for doc in normalized_docs
    ]

    # ── Stemming ──
    stemmer = PorterStemmer()
    stemmed_docs = [
        [stemmer.stem(token) for token in doc]
        for doc in filtered_docs
    ]

    # ── Lematización ──
    lemmatizer = WordNetLemmatizer()
    lemmatized_docs = [
        [lemmatizer.lemmatize(token) for token in doc]
        for doc in filtered_docs
    ]

    # ── Vocabularios ──
    all_stems = [token for doc in stemmed_docs for token in doc]
    all_lemmas = [token for doc in lemmatized_docs for token in doc]
    vocab_stems = set(all_stems)
    vocab_lemmas = set(all_lemmas)

    return {
        "tokenized": tokenized_docs,
        "normalized": normalized_docs,
        "filtered": filtered_docs,
        "stemmed": stemmed_docs,
        "lemmatized": lemmatized_docs,
        "vocab_stems": vocab_stems,
        "vocab_lemmas": vocab_lemmas
    }

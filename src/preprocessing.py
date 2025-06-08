import nltk
import pandas as pd
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_documents(documents, return_type='df'):
    """
    Preprocess a list of documents: lowercase, tokenize, remove stopwords, lemmatize.

    Args:
        documents (List[str]): List of raw document texts.
        return_type (str): 'df' to return DataFrame, 'tokens' to return tokenized corpus.

    Returns:
        pd.DataFrame: DataFrame with original and preprocessed documents.
    """
    df = pd.DataFrame(documents, columns=['document'])
    
    # Lowercase and tokenize with regex
    df['regex_tokens'] = df['document'].str.lower().apply(
        lambda text: regexp_tokenize(text, pattern=r'\w[a-z]+')
    )

    # Remove stopwords
    df['no_stopwords'] = df['regex_tokens'].apply(remove_stopwords)

    # Lemmatize
    df['lemmas'] = df['no_stopwords'].apply(lemmatize_tokens)

    # Join tokens into a single string
    df['prep_doc'] = df['lemmas'].str.join(' ')

    if return_type == 'tokens':
        return df['lemmas'].tolist()
    else:
        return df[['document', 'prep_doc']]

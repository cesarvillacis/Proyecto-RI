import ir_datasets
import os
import pickle

DATASET_NAME = "beir/cqadupstack/gaming"
CACHE_FILE = "src/beir_gaming_cached.pkl"

def load_beir_documents(limit=None):
    """
    Loads documents from the BEIR 'cqadupstack/gaming' dataset with caching.
    
    Args:
        limit (int or None): Number of documents to return. If None, returns all.

    Returns:
        List[str]: A list of document texts.
    """
    # Check if the cached version exists
    if os.path.exists(CACHE_FILE):
        print("Loading dataset from cache...")
        with open(CACHE_FILE, "rb") as f:
            documents = pickle.load(f)
    else:
        print("Loading dataset from ir_datasets and caching...")
        dataset = ir_datasets.load(DATASET_NAME)
        documents = [doc.text for doc in dataset.docs_iter()]
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(documents, f)

    if limit is not None:
        return documents[:limit]
    return documents

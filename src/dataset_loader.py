import ir_datasets

def load_beir_documents(limit=None):
    """
    Loads documents from the BEIR 'cqadupstack/gaming' dataset.
    
    Args:
        limit (int or None): Number of documents to load. If None, loads all documents.

    Returns:
        List[str]: A list of document texts.
    """
    dataset = ir_datasets.load("beir/cqadupstack/gaming")
    documents = [doc.text for doc in dataset.docs_iter()]

    if limit is not None:
        documents = documents[:limit]

    return documents

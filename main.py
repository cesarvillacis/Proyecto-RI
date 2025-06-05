from src.dataset_loader import load_beir_documents
from src.preprocessing import preprocess_documents

# Load documents from the BEIR dataset
documents = load_beir_documents(limit=1)  # Change limit as needed

# Preprocess the documents
result_df = preprocess_documents(documents)

# Display result
print(result_df)

import ir_datasets
from preprocesamiento import preprocesamiento

# Cargar documentos del dataset BEIR
dataset = ir_datasets.load("beir/cqadupstack/gaming")
docs_prueba = [doc.text for doc in dataset.docs_iter()]

# Ejecutar preprocesamiento sobre los documentos
resultado = preprocesamiento(docs_prueba[:1])  # solo uno para prueba rápida

# Mostrar resultado
print(resultado)

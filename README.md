# Sistema de Recuperación de Información con TF-IDF y BM25

Este proyecto implementa un sistema de recuperación de documentos utilizando dos métodos de procesamiento de texto:**TF-IDF** y **BM25**. Se utiliza el dataset `beir/cqadupstack/gaming`, proporcionado por la librería `ir_datasets`.

## Estructura del Proyecto

```
src/
├── dataset_loader.py       # Carga y caching de documentos y queries del dataset
├── preprocessing.py        # Preprocesamiento: tokenización, stopwords, lematización
├── search_engine.py        # Modelos de recuperación: similitud coseno y BM25
├── perf_metrics.py         # Métricas: tiempo, precisión, recall, average precision

main.py                     # Script principal de ejecución y demostración
README.md                   # Instrucciones y documentación
requirements.txt            # Lista de dependencias
```


## Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/cesarvillacis/Proyecto-RI.git
   cd proyecto-ir

2. pip install -r requirements.txt

## Requisitos
1. Python 3.8 o superior

2. Librerías:
    pandas
    numpy
    scikit-learn
    nltk  
    ir_datasets
    rank_bm25
    time

## Descarga de recursos necesarios de NLTK
Los mismos se descargan automáticamente, pero se los puede obtener mediante
  nltk.download('punkt', quiet=True)
  nltk.download('stopwords', quiet=True)
  nltk.download('wordnet', quiet=True)

## Ejecución
El sistema se ejecuta desde el archivo **main.py** o con **python main.py**
Dentro del script se puede:
  Cargar documentos del dataset BEIR.
  Preprocesar documentos (limpieza, tokenización, lematización).
  Construir representaciones TF-IDF o BM25.
  Ingresar una consulta de prueba.
  Obtener los documentos más relevantes usando similitud coseno o puntajes BM25.

## Métricas de Evaluación
El proyecto incluye funciones para calcular:
  Tiempo de ejecución de cada método (@execute_time)
  Precisión y recall en el top-k
  Precisión promedio (Average Precision)


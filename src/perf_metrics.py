import time
from functools import wraps

def execute_time(func):
    """
    Wrapper que mide el tiempo de ejecución de una función.

    Parámetros:
        func (Callable): Función a modificar con wrapper

    Retorna:
        Callable: Función modificada con wrapper que retorna una tupla 
        con el resultado original y el tiempo de ejecución en segundos.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

def precision_recall_at_k(y_true, y_pred, k):
    """
    Calcula la precisión y el recall en el top-k de documentos recuperados.

    Parámetros:
        y_true (list[str]): Lista de documentos relevantes.
        y_pred (list[str]): Lista de documentos recuperados ordenados por relevancia.
        k (int): Número de documentos a considerar desde el top.

    Retorna:
        tuple:
            - float: Precisión en el top-k.
            - float: Recall en el top-k.
    """
    top_k = y_pred[:k]
    true_positives = len(set(top_k) & set(y_true))

    precision = true_positives / k if k > 0 else 0
    recall = true_positives / len(y_true) if y_true else 0

    return precision, recall

def average_precision(y_true, y_pred):
    """
    Calcula la Precisión Promedio (Average Precision, AP) para una consulta.

    Parámetros:
        y_true (list[str]): Lista de documentos relevantes.
        y_pred (list[str]): Lista de documentos recuperados ordenados por relevancia.

    Retorna:
        float: Precisión promedio considerando la posición de cada documento relevante.
    """
    if not y_true:
        return 0.0

    precision_sum = 0.0
    num_relevant = 0

    # Acumula precisión solo en las posiciones donde hay documentos relevantes
    for i, doc in enumerate(y_pred):
        if doc in y_true:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    return precision_sum / len(y_true) if len(y_true) > 0 else 0.0

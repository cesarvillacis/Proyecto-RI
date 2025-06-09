import time
from functools import wraps

def execute_time(func):
    """
    Wrapper para medir el tiempo de ejecución de una función.
    Recibe como parámetros la función a ejecutar y sus argumentos.
    Retorna el resultado de la función y el tiempo de ejecución en segundos.
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
    Función que calcula la precisión y el recall de las predicciones.
    Recibe como parámetros los documentos relevantes, los documentos recuperados y el valor de k.
    Retorna la precisión y el recall.
    """
    top_k = y_pred[:k]
    true_positives = len(set(top_k) & set(y_true))

    precision = true_positives / k if k > 0 else 0
    recall = true_positives / len(y_true) if y_true else 0

    return precision, recall

def average_precision(y_true, y_pred):
    """
    Función que calcula la precisión promedio de las predicciones.
    Recibe como parámetros los documentos relevantes y los documentos recuperados.
    Retorna la precisión promedio.
    """
    if not y_true:
        return 0.0

    precision_sum = 0.0
    num_relevant = 0

    for i, doc in enumerate(y_pred):
        if doc in y_true:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)

    return precision_sum / len(y_true) if len(y_true) > 0 else 0.0

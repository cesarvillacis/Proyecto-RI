from sklearn.datasets import fetch_20newsgroups
from preprocesamiento import preprocesamiento

# Carga los datos completos, pero solo selecciona uno para probar
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# Solo tomar el primer documento para prueba rápida
docs_prueba = [newsgroups.data[0]]

resultado = preprocesamiento(docs_prueba)

print(resultado)

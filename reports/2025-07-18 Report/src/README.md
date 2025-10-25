# Módulos del Proyecto

Este directorio contiene las funciones reutilizables para el análisis de datos del proyecto.

## Archivos

- **preprocesamiento.py**: Limpieza de datos, extracción de parámetros, normalización.
- **visualizacion.py**: Gráficos de dispersión y clusters.
- **clustering.py**: Aplicación de t-SNE/ UMAP y DBSCAN para reducción de dimensionalidad y clustering.
- **codificacion.py**: Codificación de parámetros categóricos y procesamiento de un diccionario de DataFrames con valores categóricos en memoria, codificándolos.

Puedes importar las funciones así:

```python
from src.preprocesamiento import extract_params, extract_trace_parameters
```

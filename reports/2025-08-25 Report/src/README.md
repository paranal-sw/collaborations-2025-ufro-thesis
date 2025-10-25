# Análisis de Parámetros del Instrumento MATISSE

Este repositorio contiene los módulos y scripts necesarios para el análisis de los parámetros de trazas del instrumento **MATISSE**.

---

## Estructura del proyecto

- **`preprocesamiento.py`**: Limpieza y extracción de parámetros.
- **`codificacion.py`:** Codificación de variables categóricas.
- **`visualizacion.py:`** Gráficos comparativos de parámetros.
- **`clustering.py`:** Reducción de dimensionalidad y agrupamiento (DBSCAN + t-SNE/UMAP).

---

## Funciones principales

### `preprocesamiento.py`

#### `extract_params(df_original)`

- **Propósito:** Limpia y extrae parámetros de un DataFrame de registros. Convierte valores a tipos numéricos y separa texto en columnas categóricas.
- **Parámetros:**
  - `df_original (pd.DataFrame)`  DataFrame con registros originales.
- **Retorno:**
  - `pd.DataFrame` con columnas divididas en:
    - `numval_*`  parámetros numéricos.
    - `strval_*`  parámetros categóricos.

---

#### `extract_trace_parameters(df_meta, df_traces, extract_params, tpl_id)`

- **Propósito:** Filtra y extrae los parámetros de traza para un `TPL_ID` específico, organizando los datos por `trace_id`.
- **Parámetros:**
  - `df_meta (pd.DataFrame)`  Metadatos de registros.
  - `df_traces (pd.DataFrame)` Registros de trazas.
  - `extract_params (function)`  Función de limpieza (`extract_params`).
  - `tpl_id (str)`  Identificador de TPL a procesar.
- **Retorno:**
  - `pd.DataFrame` con parámetros numéricos extraídos por traza.

---

### `codificacion.py`

#### `codificar_categoricas_por_columna(df_categorico)`

- **Propósito:** Convierte valores de texto en representaciones numéricas (enteros), asignados en orden alfabético.
- **Parámetros:**
  - `df_categorico (pd.DataFrame)`  Columnas categóricas.
- **Retorno:**
  - `(pd.DataFrame, dict)`
    - DataFrame con columnas codificadas.
    - Diccionario con mapeos de codificación.

---

#### `procesar_strval_in_memory(df_dict, diccionario_json_path=None)`

- **Propósito:** Procesa un diccionario de DataFrames, codificando sus valores de texto.Opcionalmente guarda el diccionario de codificación en un archivo JSON.
- **Parámetros:**
  - `df_dict (dict[str, pd.DataFrame])`  Diccionario de DataFrames.
  - `diccionario_json_path (str, opcional)`  Ruta para guardar codificación.
- **Retorno:**
  - `(dict, dict)`
    - Diccionario de DataFrames codificados.
    - Diccionario con codificadores globales.

---

### `visualizacion.py`

#### `graficos_por_columna(df_sin_error, df_con_error, nombre_archivo=None, tipo="dispersion", log_y=False, use_log=True)`

- **Propósito:** Genera gráficos comparativos de parámetros entre datos **sin error** y **con error**.Admite 4 tipos de gráficos:
  - `"dispersion"`  Dispersión punto a punto.
  - `"histograma"`  Histogramas de distribución.
  - `"densidad"`  Estimación de densidad (KDE).
  - `"pie"`  Diagramas de anillos concéntricos (sin error = azul, con error = rojo).
- **Parámetros:**
  - `df_sin_error (pd.DataFrame)`  Datos de observaciones sin error.
  - `df_con_error (pd.DataFrame)`  Datos con error.
  - `nombre_archivo (str, opcional)`  Texto para el título de gráficos.
  - `tipo (str)`  Tipo de gráfico a generar.
  - `log_y (bool)` Escala logarítmica en histogramas.
  - `use_log (bool)`  Escala logarítmica en conteos de gráficos de pastel.
- **Retorno:**
  - `list[str]` Lista de parámetros constantes que fueron omitidos.

---

### `clustering.py`

#### `cluster_y_con_tsne(df_numerico, nombre="", use_umap=False, dbscan_eps=None, dbscan_min_samples=None)`

- **Propósito:** Aplica **DBSCAN** sobre datos numéricos escalados y reduce dimensionalidad con **t-SNE** o **UMAP** para visualización.
- **Parámetros:**
  - `df_numerico (pd.DataFrame)` Datos numéricos limpios.
  - `nombre (str)` Nombre del dataset.
  - `use_umap (bool)` Usa UMAP en lugar de t-SNE.
  - `dbscan_eps (float, opcional)` Parámetro `eps` de DBSCAN.
  - `dbscan_min_samples (int, opcional)` `min_samples` de DBSCAN.
- **Retorno:**
  - `pd.DataFrame` con columnas: `x`, `y`, `cluster`.

---

#### `procesar_archivos_especificos(lista_archivos_ok, lista_archivos_err, carpeta_ok, carpeta_err, usar_umap=False, dbscan_eps=None, dbscan_min_samples=None)`

- **Propósito:** Procesa múltiples archivos CSV de datos OK y ERROR, aplicando clustering y reducción de dimensionalidad.
- **Retorno:**
  - `(dict, dict)` con resultados para OK y ERROR.

---

#### `plot_combined_clusters(df_ok_result, df_err_result, filename_base, use_umap=False)`

- **Propósito:** Genera un gráfico combinado de resultados de clustering para datos **OK (círculos)** y **ERROR (X)**.
- **Parámetros:**
  - `df_ok_result (pd.DataFrame)` Clusters de datos OK.
  - `df_err_result (pd.DataFrame)` Clusters de datos ERROR.
  - `filename_base (str)` Nombre base para título/archivo.
  - `use_umap (bool)` Método de reducción usado.
- **Retorno:**
  - `None` (muestra un gráfico).

---

#### `cluster_and_plot_combined(df_ok_path, df_err_path, filename_base, use_umap=False, dbscan_eps=None, dbscan_min_samples=None)`

- **Propósito:** Carga, limpia, combina y escala datos OK y ERROR; aplica DBSCAN + t-SNE/UMAP y genera un gráfico conjunto.
- **Parámetros:**
  - `df_ok_path (str)` Ruta CSV datos OK.
  - `df_err_path (str)` Ruta CSV datos ERROR.
  - `filename_base (str)` Nombre base para título/archivo.
  - `use_umap (bool)` Si True, usa UMAP en lugar de t-SNE.
  - `dbscan_eps (float, opcional)` `eps` de DBSCAN.
  - `dbscan_min_samples (int, opcional)` `min_samples` de DBSCAN.
- **Retorno:**
  - `None` (genera gráfico y exporta CSV con resultados).

---

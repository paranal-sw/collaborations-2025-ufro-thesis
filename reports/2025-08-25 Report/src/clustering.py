import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def cluster_y_con_tsne(df_numerico, nombre="", use_umap=False,
                                   dbscan_eps=None, dbscan_min_samples=None):
    """
    Aplica DBSCAN sobre los datos reales escalados y usa t-SNE o UMAP para visualización.

    Args:
        df_numerico (DataFrame): Datos numéricos limpios y normalizados.
        nombre (str): Nombre del dataset o archivo.
        use_umap (bool): Si True, usa UMAP para la visualización en lugar de t-SNE.
        dbscan_eps (float, optional): Valor de eps para DBSCAN. Si es None, se calcula dinámicamente.
        dbscan_min_samples (int, optional): Valor de min_samples para DBSCAN. Si es None, se calcula dinámicamente.

    Returns:
        DataFrame: Resultado con coordenadas 2D y etiquetas de clúster.
    """
    n_samples, n_features = df_numerico.shape
    if n_samples < 2 or n_features < 2:
        print(f"Muy pocos datos para procesar ({n_samples} muestras, {n_features} características).")
        return None

    X_scaled = StandardScaler().fit_transform(df_numerico)

    if dbscan_eps is None:
        eps_calculado = 0.5 * np.log(n_samples) / np.sqrt(n_features)
    else:
        eps_calculado = dbscan_eps

    if dbscan_min_samples is None:
        min_samples_calculado = max(5, int(np.log(n_samples)))
    else:
        min_samples_calculado = dbscan_min_samples

    print(f"DBSCAN Parameters for {nombre}: eps={eps_calculado:.3f}, min_samples={min_samples_calculado}")
    dbscan = DBSCAN(eps=eps_calculado, min_samples=min_samples_calculado)

    etiquetas = dbscan.fit_predict(X_scaled)

    if use_umap:
        umap_n_neighbors = min(15, max(2, n_samples - 1)) 
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=0.1)
        coords_2d = reducer.fit_transform(X_scaled)
        metodo = "UMAP"
    else:
        tsne_perplexity = min(30, max(5, n_samples - 1))
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate=200,
                    max_iter=1000, init='pca', random_state=42, n_jobs=-1)
        coords_2d = tsne.fit_transform(X_scaled)
        metodo = "t-SNE"

    df_resultado = pd.DataFrame(coords_2d, columns=['x', 'y'], index=df_numerico.index)
    df_resultado['cluster'] = etiquetas
        
    return df_resultado

def procesar_archivos_especificos(lista_archivos_ok, lista_archivos_err,
                                   carpeta_ok, carpeta_err, usar_umap=False,
                                   dbscan_eps=None, dbscan_min_samples=None):
    resultados_ok = {}
    resultados_err = {}

    for archivo in lista_archivos_ok:
        ruta = os.path.join(carpeta_ok, archivo)
        if not os.path.exists(ruta):
            print(f"Archivo no encontrado: {ruta}")
            continue
        try:
            df = pd.read_csv(ruta, index_col=0)
        except Exception as e:
            print(f"Error al leer {ruta}: {e}")
            continue

        df_num = df.select_dtypes(include=["number"])
        df_num = df_num.dropna(axis=1, how='all')
        df_num = df_num.loc[:, df_num.std() != 0]
        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        df_num = df_num.dropna(axis=0, how='any')

        if df_num.empty or df_num.shape[0] < 2:
            print(f"DataFrame numérico vacío o muy pequeño después de la limpieza para {archivo}.")
            continue

        plot_filename_ok = f"{os.path.splitext(archivo)[0]}_OK.png"
        resultado = cluster_y_con_tsne(df_num, nombre=f"{archivo} (OK)",
                                                   use_umap=usar_umap,
                                                   dbscan_eps=dbscan_eps,
                                                   dbscan_min_samples=dbscan_min_samples)
        if resultado is not None:
            resultado['tipo'] = 'OK'
            resultados_ok[archivo] = resultado

    for archivo in lista_archivos_err:
        ruta = os.path.join(carpeta_err, archivo)
        if not os.path.exists(ruta):
            print(f"Archivo no encontrado: {ruta}")
            continue
        try:
            df = pd.read_csv(ruta, index_col=0)
        except Exception as e:
            print(f"Error al leer {ruta}: {e}")
            continue

        df_num = df.select_dtypes(include=["number"])
        df_num = df_num.dropna(axis=1, how='all')
        df_num = df_num.loc[:, df_num.std() != 0]
        df_num = df_num.replace([np.inf, -np.inf], np.nan)
        df_num = df_num.dropna(axis=0, how='any')

        if df_num.empty or df_num.shape[0] < 2:
            print(f"DataFrame numérico vacío o muy pequeño después de la limpieza para {archivo}.")
            continue

        plot_filename_err = f"{os.path.splitext(archivo)[0]}_ERROR.png"
        resultado = cluster_y_con_tsne(df_num, nombre=f"{archivo} (ERROR)",
                                                   use_umap=usar_umap,
                                                   dbscan_eps=dbscan_eps,
                                                   dbscan_min_samples=dbscan_min_samples)
        if resultado is not None:
            resultado['tipo'] = 'ERROR'
            resultados_err[archivo] = resultado

    return resultados_ok, resultados_err

def plot_combined_clusters(df_ok_result, df_err_result, filename_base, use_umap=False):
    """
    Genera un gráfico combinado mostrando clusters de datos OK y ERROR en el mismo plot.

    Args:
        df_ok_result (pd.DataFrame): DataFrame con resultados de clustering para datos OK.
        df_err_result (pd.DataFrame): DataFrame con resultados de clustering para datos ERROR.
        filename_base (str): Nombre base del archivo para el título y para guardar el gráfico.
        use_umap (bool): Si True, indica que se usó UMAP para la reducción de dimensionalidad.
    """
    print(f"plot_combined_clusters called with: filename_base={filename_base}, use_umap={use_umap}")

    plt.figure(figsize=(14, 10))
    metodo = "UMAP" if use_umap else "t-SNE"

    # Graficar datos OK
    if df_ok_result is not None and not df_ok_result.empty:
        unique_labels_ok = sorted(set(df_ok_result['cluster']))
        cmap_ok = plt.cm.get_cmap("viridis", len(unique_labels_ok) if len(unique_labels_ok) > 0 else 1)
        for i, label in enumerate(unique_labels_ok):
            mask = df_ok_result['cluster'] == label
            label_text = f"OK Cluster {label}" if label != -1 else "OK Ruido"
            plt.scatter(df_ok_result.loc[mask, 'x'],
                        df_ok_result.loc[mask, 'y'],
                        s=80, marker='o', # Círculos para OK
                        label=label_text,
                        alpha=0.6,
                        color=cmap_ok(i % cmap_ok.N))
    else:
        print(f"No hay datos OK para graficar para {filename_base}.")

    # Graficar datos ERROR
    if df_err_result is not None and not df_err_result.empty:
        unique_labels_err = sorted(set(df_err_result['cluster']))
        cmap_err = plt.cm.get_cmap("plasma", len(unique_labels_err) if len(unique_labels_err) > 0 else 1) # Diferente mapa de colores
        for i, label in enumerate(unique_labels_err):
            mask = df_err_result['cluster'] == label
            label_text = f"ERROR Cluster {label}" if label != -1 else "ERROR Ruido"
            plt.scatter(df_err_result.loc[mask, 'x'],
                        df_err_result.loc[mask, 'y'],
                        s=100, marker='X', # 'X' para ERROR
                        label=label_text,
                        alpha=0.7,
                        color=cmap_err(i % cmap_err.N),
                        edgecolors='black', linewidth=0.5)
    else:
        print(f"No hay datos ERROR para graficar para {filename_base}.")

    plt.title(f"Clustering Combinado: {filename_base} – {metodo}\n(O = OK, X = ERROR)", fontsize=16)
    plt.xlabel(f"{metodo} 1", fontsize=13)
    plt.ylabel(f"{metodo} 2", fontsize=13)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, ncol=1)
    plt.tight_layout()
    plt.grid(alpha=0.4)
    plt.show()

def cluster_and_plot_combined(df_ok_path, df_err_path, filename_base, use_umap=False,
                              dbscan_eps=None, dbscan_min_samples=None):
    """
    Carga, combina, escala, aplica reducción de dimensionalidad y clustering,
    y luego grafica los datos OK y ERROR en un solo plot con escala consistente.

    Args:
        df_ok_path (str): Ruta completa al archivo CSV de datos OK.
        df_err_path (str): Ruta completa al archivo CSV de datos ERROR.
        filename_base (str): Nombre base del archivo para el título y para guardar el gráfico.
        use_umap (bool): Si True, usa UMAP para la visualización en lugar de t-SNE.
        dbscan_eps (float, optional): Valor de eps para DBSCAN. Si es None, se calcula dinámicamente.
        dbscan_min_samples (int, optional): Valor de min_samples para DBSCAN. Si es None, se calcula dinámicamente.
    """
    print(f"\n--- Procesando y combinando datos para {filename_base} ---")

    df_ok_num = None
    df_err_num = None

    if os.path.exists(df_ok_path):
        try:
            df_ok_raw = pd.read_csv(df_ok_path, index_col=0)
            df_ok_num = df_ok_raw.select_dtypes(include=["number"])
            df_ok_num = df_ok_num.dropna(axis=1, how='all')
            df_ok_num = df_ok_num.loc[:, df_ok_num.std() != 0]
            df_ok_num = df_ok_num.replace([np.inf, -np.inf], np.nan)
            if df_ok_num.empty or df_ok_num.shape[0] < 1:
                print(f"Datos OK vacíos o muy pequeños después de la limpieza de columnas para {filename_base}.")
                df_ok_num = None
            else:
                df_ok_num['__type__'] = 'OK'
        except Exception as e:
            print(f"Error al leer o procesar datos OK de {df_ok_path}: {e}")
            df_ok_num = None
    else:
        print(f"Archivo OK no encontrado: {df_ok_path}")
        df_ok_num = None

    if os.path.exists(df_err_path):
        try:
            df_err_raw = pd.read_csv(df_err_path, index_col=0)
            df_err_num = df_err_raw.select_dtypes(include=["number"])
            df_err_num = df_err_num.dropna(axis=1, how='all')
            df_err_num = df_err_num.loc[:, df_err_num.std() != 0]
            df_err_num = df_err_num.replace([np.inf, -np.inf], np.nan)
            if df_err_num.empty or df_err_num.shape[0] < 1:
                print(f"Datos ERROR vacíos o muy pequeños después de la limpieza de columnas para {filename_base}.")
                df_err_num = None
            else:
                df_err_num['__type__'] = 'ERROR'
        except Exception as e:
            print(f"Error al leer o procesar datos ERROR de {df_err_path}: {e}")
            df_err_num = None
    else:
        print(f"Archivo ERROR no encontrado: {df_err_path}")
        df_err_num = None

    combined_df = pd.DataFrame()
    if df_ok_num is not None and df_err_num is not None:
        numeric_cols_ok = [col for col in df_ok_num.columns if col != '__type__']
        numeric_cols_err = [col for col in df_err_num.columns if col != '__type__']
        all_numeric_cols = sorted(list(set(numeric_cols_ok + numeric_cols_err)))

        df_ok_aligned = df_ok_num.reindex(columns=all_numeric_cols + ['__type__'], fill_value=np.nan)
        df_err_aligned = df_err_num.reindex(columns=all_numeric_cols + ['__type__'], fill_value=np.nan)

        combined_df = pd.concat([df_ok_aligned, df_err_aligned], ignore_index=False)
    elif df_ok_num is not None:
        combined_df = df_ok_num
    elif df_err_num is not None:
        combined_df = df_err_num

    if combined_df.empty or combined_df.shape[0] < 2:
        print(f"DataFrame combinado vacío o muy pequeño después de la unión para {filename_base}.")
        return

    initial_rows = combined_df.shape[0]
    
    types = combined_df['__type__']
    combined_df_numeric = combined_df.drop(columns=['__type__'])

    combined_df_numeric_imputed = combined_df_numeric.fillna(0) 
  
    if combined_df_numeric_imputed.empty or combined_df_numeric_imputed.shape[0] < 2:
        print(f"DataFrame combinado muy pequeño después de la imputación para {filename_base}.")
        return

    X_scaled_combined = StandardScaler().fit_transform(combined_df_numeric_imputed)
    n_samples_combined = combined_df_numeric_imputed.shape[0]
    n_features_combined = combined_df_numeric_imputed.shape[1]

    if dbscan_eps is None:
        if n_features_combined == 0:
            print(f"No hay características numéricas para DBSCAN en {filename_base}.")
            return
        eps_calculated = 0.5 * np.log(n_samples_combined) / np.sqrt(n_features_combined)
    else:
        eps_calculated = dbscan_eps

    if dbscan_min_samples is None:
        min_samples_calculated = max(2, int(np.log(n_samples_combined)))
    else:
        min_samples_calculated = dbscan_min_samples

    print(f"DBSCAN Parameters for Combined Data: eps={eps_calculated:.3f}, min_samples={min_samples_calculated}")
    dbscan = DBSCAN(eps=eps_calculated, min_samples=min_samples_calculated)
    combined_clusters = dbscan.fit_predict(X_scaled_combined)

    metodo = "UMAP"
    if use_umap:
        umap_n_neighbors = min(15, max(2, n_samples_combined - 1)) 
        reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=0.1)
        combined_coords_2d = reducer.fit_transform(X_scaled_combined)
    else:
        metodo = "t-SNE"
        tsne_perplexity = min(30, max(5, n_samples_combined - 1))
        if n_samples_combined <= 1:
            print(f"No hay suficientes muestras para t-SNE en {filename_base} después de la limpieza.")
            return

        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate=200,
                    max_iter=1000, init='pca', random_state=42, n_jobs=-1)
        combined_coords_2d = tsne.fit_transform(X_scaled_combined)

    combined_result_df = pd.DataFrame(combined_coords_2d, columns=['x', 'y'], index=combined_df_numeric_imputed.index)
    combined_result_df['cluster'] = combined_clusters
    combined_result_df['type'] = types 
    combined_result_df.to_csv(f'{filename_base}_combined_results__{metodo.lower()}.csv', index=True)

    df_ok_result_for_plot = combined_result_df[combined_result_df['type'] == 'OK'].copy()
    df_err_result_for_plot = combined_result_df[combined_result_df['type'] == 'ERROR'].copy()

    # Graficar los resultados combinados
    plot_combined_clusters(df_ok_result_for_plot, df_err_result_for_plot, f"{filename_base}_{metodo.lower()}", use_umap=use_umap)

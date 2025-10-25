import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import DBSCAN

def aplicar_clustering(df_enriched, excluir_cols=None, eps=2, min_samples=5, random_state=42):
    """
    Aplica reducción de dimensionalidad (t-SNE, UMAP) y clustering (DBSCAN)
    sobre el dataset enriquecido.

    Parámetros:
        df_enriched : DataFrame con parámetros + metadatos
        excluir_cols : lista de columnas a excluir (ej. ["ERROR","TPL_ID","SECONDS"])
        eps : float, parámetro de DBSCAN
        min_samples : int, parámetro de DBSCAN
        random_state : int, semilla para reproducibilidad

    Retorna:
        df_enriched : DataFrame con columnas nuevas:
            - tSNE_1, tSNE_2
            - UMAP_1, UMAP_2
            - Cluster_DBSCAN
    """

    if excluir_cols is None:
        excluir_cols = ["ERROR", "TPL_ID", "SECONDS"]

    X = df_enriched.drop(columns=[c for c in excluir_cols if c in df_enriched.columns], errors="ignore")
    X = X.select_dtypes(include=np.number)

    X = X.loc[:, X.isna().mean() < 0.8]
    X = X.loc[X.isna().mean(axis=1) < 0.9]

    if X.empty:
        raise ValueError("No hay datos suficientes para clustering después del filtrado.")

    X_scaled = StandardScaler().fit_transform(X.fillna(0))

    # t-SNE
    tsne_emb = TSNE(n_components=2, random_state=random_state).fit_transform(X_scaled)
    df_enriched.loc[X.index, "tSNE_1"] = tsne_emb[:, 0]
    df_enriched.loc[X.index, "tSNE_2"] = tsne_emb[:, 1]

    # UMAP
    umap_emb = umap.UMAP(n_components=2, random_state=random_state).fit_transform(X_scaled)
    df_enriched.loc[X.index, "UMAP_1"] = umap_emb[:, 0]
    df_enriched.loc[X.index, "UMAP_2"] = umap_emb[:, 1]

    # DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    df_enriched.loc[X.index, "Cluster_DBSCAN"] = labels

    return df_enriched

def construir_dataset_enriquecido(
    df_meta: pd.DataFrame,
    tpl_params_dict: dict,
    codified_strval_dfs: dict
) -> pd.DataFrame:
    """
    Construye un DataFrame enriquecido combinando df_meta con parámetros numéricos
    y categóricos codificados. No separa en error/ok, trabaja todo en un solo dataset.
    """
    df_enriched = df_meta.copy()

    if "TRACE_ID" in df_enriched.columns:
        df_enriched = df_enriched.set_index("TRACE_ID")

    if tpl_params_dict:
        df_params = pd.concat(
            {tpl_id: df for tpl_id, df in tpl_params_dict.items()},
            axis=1
        )
        df_params.columns = [
            f"NUM_{tpl}.{col}" for tpl, col in df_params.columns.to_flat_index()
        ]
        df_enriched = df_enriched.join(df_params, how="left")

    if codified_strval_dfs:
        df_cats = pd.concat(
            {tpl_id: df for tpl_id, df in codified_strval_dfs.items()},
            axis=1
        )
        df_cats.columns = [
            f"CAT_{tpl}.{col}" for tpl, col in df_cats.columns.to_flat_index()
        ]
        df_enriched = df_enriched.join(df_cats, how="left")

    df_enriched = df_enriched.loc[:, ~df_enriched.columns.duplicated()]

    return df_enriched

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import DBSCAN, KMeans

def clustering_completo(
    df_enriched,
    excluir_cols=None,
    eps=2,
    min_samples=5,
    kmeans_n_clusters=5,
    random_state=42
):
    """
    Pipeline completo de reducción de dimensionalidad y clustering.

    Retorna:
        df_enriched: DataFrame con columnas de clusters
        X_tsne: embeddings t-SNE
        X_umap: embeddings UMAP
    """
    if excluir_cols is None:
        excluir_cols = ["ERROR", "TPL_ID", "SECONDS"]

    X = df_enriched.drop(columns=[c for c in excluir_cols if c in df_enriched.columns], errors="ignore")
    X = X.select_dtypes(include=np.number)

    X = X.loc[:, X.isna().mean() < 0.8]
    X = X.loc[X.isna().mean(axis=1) < 0.9]

    if X.empty:
        raise ValueError("No hay datos suficientes para clustering después del filtrado.")

    X_scaled = StandardScaler().fit_transform(X.fillna(0))

    # t-SNE
    X_tsne = TSNE(n_components=2, random_state=random_state, init='pca').fit_transform(X_scaled)
    df_enriched.loc[X.index, "tSNE_1"] = X_tsne[:,0]
    df_enriched.loc[X.index, "tSNE_2"] = X_tsne[:,1]

    # UMAP
    X_umap = umap.UMAP(n_components=2, random_state=random_state, min_dist=0.1).fit_transform(X_scaled)
    df_enriched.loc[X.index, "UMAP_1"] = X_umap[:,0]
    df_enriched.loc[X.index, "UMAP_2"] = X_umap[:,1]

    # DBSCAN sobre t-SNE
    df_enriched.loc[X.index, "Cluster_DBSCAN_TSNE"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_tsne)

    # DBSCAN sobre UMAP
    df_enriched.loc[X.index, "Cluster_DBSCAN_UMAP"] = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_umap)

    # KMeans sobre t-SNE
    df_enriched.loc[X.index, "Cluster_KMeans_TSNE"] = KMeans(n_clusters=kmeans_n_clusters, random_state=random_state).fit_predict(X_tsne)

    # KMeans sobre UMAP
    df_enriched.loc[X.index, "Cluster_KMeans_UMAP"] = KMeans(n_clusters=kmeans_n_clusters, random_state=random_state).fit_predict(X_umap)

    return df_enriched, X_tsne, X_umap

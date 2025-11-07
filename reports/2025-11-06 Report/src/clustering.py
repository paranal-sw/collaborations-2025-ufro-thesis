import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

def cluster_todos_en_conjunto(
    df_ok: pd.DataFrame | None,
    df_err: pd.DataFrame | None,
    use_umap: bool = False,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int | None = None,
    threshold_nan: float = 0.5,
):
    """
    Clustering conjunto o individual (si solo se entrega df_ok o df_err).
    Maneja automáticamente casos con pocos parámetros, muestras o columnas constantes.
    """

    # --- Caso 1: solo un DataFrame (por ejemplo, solo errores) ---
    if df_ok is None or df_err is None:
        df = df_ok if df_ok is not None else df_err
        origen = np.array(["OK"] * len(df)) if df_ok is not None else np.array(["ERR"] * len(df))

        df = df.apply(pd.to_numeric, errors="coerce").copy()

        # Filtrar columnas y filas con exceso de NaN
        col_mask = df.isna().mean() < threshold_nan
        df = df.loc[:, col_mask]
        row_mask = df.isna().mean(axis=1) < threshold_nan
        df = df.loc[row_mask, :]
        origen = origen[row_mask.values]

        if df.empty or df.shape[1] == 0:
            print("Dataset vacío tras limpieza (solo un origen). Se omite.")
            return pd.DataFrame(), {}

        # Quitar columnas constantes
        nunique = df.nunique(dropna=True)
        df = df.loc[:, nunique[nunique >= 2].index]
        if df.shape[1] == 0:
            print(" Todas las columnas son constantes tras limpieza. Se omite este TPL_ID.")
            return pd.DataFrame(), {}

        X = SimpleImputer(strategy="median").fit_transform(df)
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

        n = X.shape[0]
        p = X.shape[1]

        if p < 2:
            print(f"Solo {p} parámetro(s); se usará proyección trivial en eje X.")
            emb = np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])

        elif n < 5:
            print(f"Solo {n} muestras; se usará proyección directa de los dos primeros parámetros.")
            if p >= 2:
                emb = X[:, :2]
            else:
                emb = np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])

        else:
            if use_umap and _HAS_UMAP:
                emb = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
            else:
                perplexity = int(np.clip((n - 1)//3, 2, max(2, n - 1)))
                emb = TSNE(
                    n_components=2, learning_rate="auto", init="pca",
                    perplexity=perplexity, random_state=42
                ).fit_transform(X)

        # --- Clustering con DBSCAN ---
        if dbscan_min_samples is None:
            dbscan_min_samples = int(np.clip(np.log2(max(n, 2)), 3, 20))
        if dbscan_eps is None:
            rng = np.ptp(emb, axis=0).mean()
            dbscan_eps = max(0.05 * rng, 1e-3)

        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(emb)

        df_res = pd.DataFrame({
            "x": emb[:, 0],
            "y": emb[:, 1],
            "cluster": labels,
            "origen": origen,
        })

        return df_res, {
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "n": n,
            "p": p,
        }

    # --- Caso 2: flujo original OK vs ERR ---
    ok = df_ok.apply(pd.to_numeric, errors="coerce").copy()
    er = df_err.apply(pd.to_numeric, errors="coerce").copy()

    cols = sorted(list(set(ok.columns) & set(er.columns)))
    if not cols:
        print(" No hay columnas comunes entre OK y ERR. Se omite.")
        return pd.DataFrame(), {}
    ok, er = ok[cols], er[cols]

    both = pd.concat([ok, er], axis=0, ignore_index=True)
    origen = np.array(["OK"] * len(ok) + ["ERR"] * len(er))

    col_mask = both.isna().mean() < threshold_nan
    both = both.loc[:, col_mask]
    row_mask = both.isna().mean(axis=1) < threshold_nan
    both = both.loc[row_mask, :]
    origen = origen[row_mask.values]

    if both.empty or both.shape[1] == 0:
        print("Dataset conjunto vacío tras limpieza. Se omite.")
        return pd.DataFrame(), {}

    nunique = both.nunique(dropna=True)
    both = both.loc[:, nunique[nunique >= 2].index]
    if both.shape[1] == 0:
        print(" Todas las columnas son constantes tras limpieza. Se omite este TPL_ID.")
        return pd.DataFrame(), {}

    X = SimpleImputer(strategy="median").fit_transform(both)
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0)

    n = X.shape[0]
    p = X.shape[1]

    if p < 2:
        print(f"Solo {p} parámetro(s); se usará proyección trivial en eje X.")
        emb = np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
    elif n < 5:
        print(f" Solo {n} muestras; se usará proyección directa de los dos primeros parámetros.")
        if p >= 2:
            emb = X[:, :2]
        else:
            emb = np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
    else:
        if use_umap and _HAS_UMAP:
            emb = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        else:
            perplexity = int(np.clip((n - 1)//3, 2, max(2, n - 1)))
            emb = TSNE(
                n_components=2, learning_rate="auto", init="pca",
                perplexity=perplexity, random_state=42
            ).fit_transform(X)

    if dbscan_min_samples is None:
        dbscan_min_samples = int(np.clip(np.log2(max(n, 2)), 3, 20))
    if dbscan_eps is None:
        rng = np.ptp(emb, axis=0).mean()
        dbscan_eps = max(0.05*rng, 1e-3)

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(emb)

    df_res = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "cluster": labels,
        "origen": origen,
    })

    return df_res, {
        "eps": dbscan_eps,
        "min_samples": dbscan_min_samples,
        "n": n,
        "p": p,
    }


def plot_clusters_por_cluster_y_origen(
    df_res,
    titulo="Clusters (color=cluster, marcador=origen)",
    cmap_name="tab20",
    alpha_ok=0.55,
    alpha_err=0.8,
    s_ok=28,
    s_err=34,
    show_legend=True,
    max_legend_clusters=20,
):
    """
    Visualiza clusters detectados por DBSCAN.
    Si hay columnas:
      - x, y        (embedding 2D)
      - cluster     (labels DBSCAN)
      - origen      ("OK" o "ERR")
    Modo automático:
      - OK y ERR → círculos vs X
      - Solo ERR → puntos sólidos
      - Solo OK  → círculos huecos
    """

    clusters = np.sort(df_res["cluster"].unique())
    cmap = plt.get_cmap(cmap_name, max(len(clusters), 1))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(clusters)}

    plt.figure(figsize=(8.2, 6.8))

    origines_presentes = df_res["origen"].unique()
    tiene_ok = "OK" in origines_presentes
    tiene_err = "ERR" in origines_presentes

    for i, c in enumerate(clusters):
        sub = df_res[df_res["cluster"] == c]
        col = color_map[c]

        # --- Modo OK + ERR ---
        if tiene_ok and tiene_err:
            sub_ok = sub[sub["origen"] == "OK"]
            sub_err = sub[sub["origen"] == "ERR"]

            if not sub_ok.empty:
                plt.scatter(
                    sub_ok["x"], sub_ok["y"],
                    s=s_ok, facecolors="none", edgecolors=col, alpha=alpha_ok,
                    label=f"cl{c}·OK" if show_legend and i < max_legend_clusters else None
                )
            if not sub_err.empty:
                plt.scatter(
                    sub_err["x"], sub_err["y"],
                    s=s_err, c=[col], marker="x", alpha=alpha_err,
                    label=f"cl{c}·ERR" if show_legend and i < max_legend_clusters else None
                )

        # --- Solo ERR ---
        elif tiene_err:
            plt.scatter(
                sub["x"], sub["y"],
                s=s_err, c=[col], alpha=alpha_err, edgecolors="none",
                label=f"Cluster {c}" if show_legend and i < max_legend_clusters else None
            )

        # --- Solo OK ---
        elif tiene_ok:
            plt.scatter(
                sub["x"], sub["y"],
                s=s_ok, facecolors="none", edgecolors=col, alpha=alpha_ok,
                label=f"Cluster {c}" if show_legend and i < max_legend_clusters else None
            )

    plt.title(titulo)
    if show_legend:
        plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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
    df_ok: pd.DataFrame,
    df_err: pd.DataFrame,
    use_umap: bool = False,
    dbscan_eps: float | None = None,
    dbscan_min_samples: int | None = None,
    threshold_nan: float = 0.5,
):
    ok = df_ok.apply(pd.to_numeric, errors="coerce").copy()
    er = df_err.apply(pd.to_numeric, errors="coerce").copy()

    cols = sorted(list(set(ok.columns) & set(er.columns)))
    if not cols:
        raise ValueError("No hay columnas comunes entre OK y ERR.")
    ok, er = ok[cols], er[cols]

    both = pd.concat([ok, er], axis=0, ignore_index=True)
    origen = np.array(["OK"]*len(ok) + ["ERR"]*len(er))

    col_mask = both.isna().mean() < threshold_nan
    both = both.loc[:, col_mask]
    row_mask = both.isna().mean(axis=1) < threshold_nan
    both = both.loc[row_mask, :]
    origen = origen[row_mask.values]

    if both.empty or both.shape[1] == 0:
        raise ValueError("Dataset conjunto vacío tras limpieza.")

    nunique = both.nunique(dropna=True)
    both = both.loc[:, nunique[nunique >= 2].index]
    if both.shape[1] == 0:
        raise ValueError("Todas las columnas quedaron constantes tras limpieza.")

    X = SimpleImputer(strategy="median").fit_transform(both)
    X = StandardScaler().fit_transform(X)

    n = X.shape[0]
    if use_umap and _HAS_UMAP:
        emb = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
    else:
        perplexity = int(np.clip((n - 1)//3, 5, 30))
        emb = TSNE(n_components=2, learning_rate="auto", init="pca",
                   perplexity=perplexity, random_state=42).fit_transform(X)

    if dbscan_min_samples is None:
        dbscan_min_samples = int(np.clip(np.log2(max(n, 2)), 3, 20))
    if dbscan_eps is None:
        rng = np.ptp(emb, axis=0).mean()
        dbscan_eps = max(0.05*rng, 1e-3)
    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(emb)

    df_res = pd.DataFrame({
        "x": emb[:,0],
        "y": emb[:,1],
        "cluster": labels,
        "origen": origen,
    })
    return df_res, {"eps": dbscan_eps, "min_samples": dbscan_min_samples, "n": n, "p": X.shape[1]}

import numpy as np
import matplotlib.pyplot as plt

def plot_clusters_por_cluster_y_origen(
    df_res,
    titulo="Clusters (color=cluster, marcador=origen)",
    cmap_name="tab20",   # o "hsv" si tienes muchos clusters
    alpha_ok=0.55,
    alpha_err=0.75,
    s_ok=28,
    s_err=34,
    show_legend=True,
    max_legend_clusters=20,  
):
    """
    Espera un DataFrame con columnas:
      - x, y        (embedding 2D)
      - cluster     (labels de DBSCAN; -1=ruido)
      - origen      ("OK" o "ERR")
    Colorea por 'cluster' y diferencia origen con marcador:
      OK = círculo (edgecolor)
      ERR = 'x'
    """
    clusters = np.sort(df_res["cluster"].unique())
    cmap = plt.get_cmap(cmap_name, max(len(clusters), 1))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(clusters)}

    plt.figure(figsize=(8.2, 6.8))

    for i, c in enumerate(clusters):
        sub = df_res[df_res["cluster"] == c]
        col = color_map[c]

        sub_ok  = sub[sub["origen"] == "OK"]
        sub_err = sub[sub["origen"] == "ERR"]

        # OK: círculo hueco del color del cluster
        if not sub_ok.empty:
            plt.scatter(
                sub_ok["x"], sub_ok["y"],
                s=s_ok, facecolors="none", edgecolors=col, alpha=alpha_ok,
                label=f"cl{c}·OK" if (show_legend and i < max_legend_clusters) else None
            )
        # ERR: X del mismo color del cluster
        if not sub_err.empty:
            plt.scatter(
                sub_err["x"], sub_err["y"],
                s=s_err, c=[col], marker="x", alpha=alpha_err,
                label=f"cl{c}·ERR" if (show_legend and i < max_legend_clusters) else None
            )

    plt.title(titulo)
    if show_legend:
        plt.legend(ncol=2, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.show()


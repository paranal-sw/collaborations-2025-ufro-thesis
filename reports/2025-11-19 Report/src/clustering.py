import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


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

    IMPORTANTE:
    - DBSCAN se aplica sobre los parámetros originales normalizados (no sobre el embedding 2D).
    - El embedding 2D se obtiene con PCA y se usa únicamente para visualización.
    """


    if df_ok is None or df_err is None:
        df = df_ok if df_ok is not None else df_err
        origen = np.array(["OK"] * len(df)) if df_ok is not None else np.array(["ERR"] * len(df))

        df = df.apply(pd.to_numeric, errors="coerce").copy()


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
            print("Todas las columnas son constantes tras limpieza. Se omite este TPL_ID.")
            return pd.DataFrame(), {}

        X = SimpleImputer(strategy="median").fit_transform(df)
        X = StandardScaler().fit_transform(X)
        X = np.nan_to_num(X, nan=0.0)

        n, p = X.shape


        if p >= 2 and n >= 2:
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(X)
        elif p >= 1 and n >= 1:

            emb = np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
            print(f"Solo {p} parámetro(s); embedding 2D degenerado (X, 0).")
        else:
            print("No hay suficientes datos para generar embedding 2D. Se omite.")
            return pd.DataFrame(), {}

        if dbscan_min_samples is None:
            dbscan_min_samples = int(np.clip(np.log2(max(n, 2)), 3, 20))

        if dbscan_eps is None:
            rng = np.ptp(X, axis=0).mean()
            dbscan_eps = max(0.3 * rng, 1e-3)

        labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X)

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

    ok = df_ok.apply(pd.to_numeric, errors="coerce").copy()
    er = df_err.apply(pd.to_numeric, errors="coerce").copy()

    cols = sorted(list(set(ok.columns) & set(er.columns)))
    if not cols:
        print("No hay columnas comunes entre OK y ERR. Se omite.")
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

    # Quitar columnas constantes
    nunique = both.nunique(dropna=True)
    both = both.loc[:, nunique[nunique >= 2].index]
    if both.shape[1] == 0:
        print("Todas las columnas son constantes tras limpieza. Se omite este TPL_ID.")
        return pd.DataFrame(), {}

    X = SimpleImputer(strategy="median").fit_transform(both)
    X = StandardScaler().fit_transform(X)
    X = np.nan_to_num(X, nan=0.0)

    n, p = X.shape

    if p >= 2 and n >= 2:
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(X)
    elif p >= 1 and n >= 1:
        emb = np.column_stack([X[:, 0], np.zeros_like(X[:, 0])])
        print(f"Solo {p} parámetro(s); embedding 2D degenerado (X, 0).")
    else:
        print("No hay suficientes datos para generar embedding 2D. Se omite.")
        return pd.DataFrame(), {}

    if dbscan_min_samples is None:
        dbscan_min_samples = int(np.clip(np.log2(max(n, 2)), 3, 20))
    if dbscan_eps is None:
        rng = np.ptp(X, axis=0).mean()
        dbscan_eps = max(0.3 * rng, 1e-3)

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(X)

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
    df_res: pd.DataFrame,
    titulo: str = "Clusters en embedding 2D (color=cluster, marcador=origen)",
    cmap_name: str = "tab20",
    alpha_ok: float = 0.55,
    alpha_err: float = 0.8,
    s_ok: int = 28,
    s_err: int = 34,
    show_legend: bool = True,
    max_legend_clusters: int = 20,
    jitter: float = 0.015,
    save_path: str | None = None,   # <-- NUEVO: ruta PNG opcional
    mostrar: bool = True,           # <-- NUEVO: mostrar o cerrar figura
):
    """
    Visualiza clusters detectados por DBSCAN en un embedding 2D (PCA).
    Requiere columnas:
      - x, y        (coordenadas del embedding 2D)
      - cluster     (labels DBSCAN)
      - origen      ("OK" o "ERR")

    Se aplica jittering ligero para evitar superposición de puntos.

    Parámetros nuevos:
    - save_path: si no es None, guarda la figura en esa ruta (PNG).
    - mostrar: si True hace plt.show(), si False cierra la figura (útil para scripts).
    """

    if df_res.empty:
        print("df_res vacío; no se grafica nada.")
        return


    df_plot = df_res.copy()
    df_plot["x_j"] = df_plot["x"] + np.random.uniform(-jitter, jitter, size=len(df_plot))
    df_plot["y_j"] = df_plot["y"] + np.random.uniform(-jitter, jitter, size=len(df_plot))

    clusters = np.sort(df_plot["cluster"].unique())
    cmap = plt.get_cmap(cmap_name, max(len(clusters), 1))
    color_map = {c: cmap(i % cmap.N) for i, c in enumerate(clusters)}

    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    plt.sca(ax)

    origines_presentes = df_plot["origen"].unique()
    tiene_ok = "OK" in origines_presentes
    tiene_err = "ERR" in origines_presentes

    for i, c in enumerate(clusters):
        sub = df_plot[df_plot["cluster"] == c]
        col = color_map[c]

        # OK + ERR
        if tiene_ok and tiene_err:
            sub_ok = sub[sub["origen"] == "OK"]
            sub_err = sub[sub["origen"] == "ERR"]

            if not sub_ok.empty:
                ax.scatter(
                    sub_ok["x_j"], sub_ok["y_j"],
                    s=s_ok, facecolors="none", edgecolors=col, alpha=alpha_ok,
                    label=f"cl{c}·OK" if show_legend and i < max_legend_clusters else None,
                )
            if not sub_err.empty:
                ax.scatter(
                    sub_err["x_j"], sub_err["y_j"],
                    s=s_err, c=[col], marker="x", alpha=alpha_err,
                    label=f"cl{c}·ERR" if show_legend and i < max_legend_clusters else None,
                )

        # Solo ERR
        elif tiene_err:
            ax.scatter(
                sub["x_j"], sub["y_j"],
                s=s_err, c=[col], alpha=alpha_err, edgecolors="none",
                label=f"Cluster {c}" if show_legend and i < max_legend_clusters else None,
            )

        # Solo OK
        elif tiene_ok:
            ax.scatter(
                sub["x_j"], sub["y_j"],
                s=s_ok, facecolors="none", edgecolors=col, alpha=alpha_ok,
                label=f"Cluster {c}" if show_legend and i < max_legend_clusters else None,
            )

    ax.set_title(titulo)
    ax.set_xlabel("Dimensión 1 (embedding PCA 2D)")
    ax.set_ylabel("Dimensión 2 (embedding PCA 2D)")
    if show_legend:
        ax.legend(ncol=2, fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Cluster guardado en: {save_path}")

    if mostrar:
        plt.show()
    else:
        plt.close(fig)

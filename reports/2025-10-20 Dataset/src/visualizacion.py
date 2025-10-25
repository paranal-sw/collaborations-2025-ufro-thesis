import matplotlib.pyplot as plt
import numpy as np


def graficos_por_columna(df_ok, df_err, columnas=None):
    """
    Genera gráficos comparativos de parámetros entre datos sin error y con error.
    Detecta automáticamente el tipo de gráfico:
      - NUM_* → Histograma
      - CAT_* → Torta
    """
    if columnas is None:
        columnas = df_ok.columns.intersection(df_err.columns)
    
    for col in columnas:
        plt.figure(figsize=(6, 4))

        # --- Numéricos: histograma ---
        if col.startswith("NUM_"):
            min_val = min(df_ok[col].min(), df_err[col].min())
            max_val = max(df_ok[col].max(), df_err[col].max())
            bins = 20
            bins = np.linspace(min_val, max_val, bins)

            plt.hist(df_ok[col].dropna(), bins=bins, alpha=0.5, 
                     color="blue", label="Sin error")
            plt.hist(df_err[col].dropna(), bins=bins, alpha=0.5, 
                     color="red", label="Con error")
            plt.title(f"Histograma - {col}")
            plt.xlabel(col)
            plt.ylabel("Frecuencia")
            plt.legend()

        #Grafico Categoricos Pie Chart
        elif col.startswith("CAT_"):
            counts_ok = df_ok[col].value_counts()
            counts_err = df_err[col].value_counts()
            all_idx = counts_ok.index.union(counts_err.index)

            counts_ok = counts_ok.reindex(all_idx, fill_value=0)
            counts_err = counts_err.reindex(all_idx, fill_value=0)

            if counts_ok.sum() + counts_err.sum() == 0:
                plt.close()
                continue

            wedges_out, _ = plt.pie(counts_ok, radius=1, labels=all_idx, 
                                    wedgeprops=dict(width=0.3, edgecolor='w'),
                                    colors=plt.cm.Blues_r(
                                        np.linspace(0.4, 1, len(all_idx))))
            wedges_in, _ = plt.pie(counts_err, radius=0.7, labels=None,
                                   wedgeprops=dict(width=0.3, edgecolor='w'),
                                   colors=plt.cm.Reds_r(
                                       np.linspace(0.4, 1, len(all_idx))))
            plt.title(f"Torta - {col}")
            plt.legend([wedges_out[0], wedges_in[0]], ["Sin error", "Con error"])

        plt.show()
        
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap

def plot_clusters(X, labels, title, ax=None, cmap="tab20", alpha=0.7, s=30):
    """
    Dibuja un scatterplot de clusters con colores diferenciados.

    Parámetros:
        X : array-like, shape (n_samples, 2)
            Embeddings 2D (t-SNE o UMAP).
        labels : array-like
            Etiquetas de cluster (int), -1 = ruido.
        title : str
            Título del gráfico.
        ax : matplotlib.axes.Axes, opcional
            Eje donde dibujar.
        cmap : str
            Colormap para clusters.
        alpha : float
            Transparencia de los puntos.
        s : int
            Tamaño de los puntos.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))

    labels = np.array(labels)
    unique_labels = np.unique(labels)

    unique_labels = [l for l in unique_labels if l != -1] + ([-1] if -1 in unique_labels else [])

    palette = sns.color_palette(cmap, n_colors=len(unique_labels))
    colors = ListedColormap(palette)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1],
                   s=s,
                   color=colors(i),
                   alpha=alpha,
                   label="Ruido" if label == -1 else f"Cluster {label}")

    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    
    return ax

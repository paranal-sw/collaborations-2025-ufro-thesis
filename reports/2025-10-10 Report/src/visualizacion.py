import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

plt.style.use('default')
plt.rcParams['font.size'] = 10


def graficos_por_columna(df_sin_error, df_con_error, nombre_archivo=None,
                         tipo="histograma",  
                         log_y=False, use_log=True):
    """
    Gráficos comparando df_sin_error vs df_con_error.
    Tipos soportados: "histograma" y "pie".
    """
    columnas_parametros = [col for col in df_sin_error.columns if col != 'param']
    parametros_constantes = []

    columnas_validas = []
    for col in columnas_parametros:
        std_ok = pd.to_numeric(df_sin_error[col], errors='coerce').std()   # CAMBIO: robusto
        std_err = pd.to_numeric(df_con_error[col], errors='coerce').std()  # CAMBIO: robusto
        if (std_ok == 0 or pd.isna(std_ok)) and (std_err == 0 or pd.isna(std_err)):
            parametros_constantes.append(col)
        else:
            columnas_validas.append(col)

    print(f"Creando gráficos tipo '{tipo}' para {len(columnas_validas)} parámetros NO constantes...")
    if parametros_constantes:
        print(f"Parámetros constantes (omitidos): {parametros_constantes}")

    n_params = len(columnas_validas)
    if n_params == 0:
        print("Todos los parámetros son constantes. No se generarán gráficos.")
        return parametros_constantes

    cols = 2 if tipo == "pie" else 3
    rows = (n_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=120)

    titulo_archivo = f" - {nombre_archivo}" if nombre_archivo else ""
    fig.suptitle(
        f'Análisis Comparativo de Parámetros{titulo_archivo}\n'
        f'Tipo de gráfico: {tipo} | {n_params} parámetros variables',
        fontsize=16, fontweight='bold', y=0.98
    )

    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = np.array([axes])

    for i, param in enumerate(columnas_validas):
        ax = axes_flat[i]
        data1 = df_sin_error[param].dropna()
        data2 = df_con_error[param].dropna()

        if tipo == "histograma":
            x1 = pd.to_numeric(data1, errors='coerce').dropna().values
            x2 = pd.to_numeric(data2, errors='coerce').dropna().values

            vals_ok = pd.Series(x1)
            vals_err = pd.Series(x2)

            uniques_total = pd.Series(np.concatenate([vals_ok.values, vals_err.values])) \
                .dropna().unique()
            low_card = len(uniques_total) <= 5  
            
            arr = x1 if x2.size == 0 else (x2 if x1.size == 0 else np.concatenate([x1, x2]))
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                ax.text(0.5, 0.5, "No finitos", ha='center', va='center')
                ax.set_title(param)
            else:
                xmin, xmax = float(np.min(arr)), float(np.max(arr))

                if xmin == xmax or low_card:
                    cats = sorted(pd.Series(uniques_total).tolist())
                    c_ok = vals_ok.value_counts().reindex(cats, fill_value=0)
                    c_err = vals_err.value_counts().reindex(cats, fill_value=0)

                    x = np.arange(len(cats))
                    width = 0.45
                    ax.bar(x - width / 2, c_ok.values, width=width, alpha=0.6, label='Sin error')
                    ax.bar(x + width / 2, c_err.values, width=width, alpha=0.6, label='Con error')
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(c) for c in cats], rotation=45, ha='right')
                    ax.set_ylabel('Frecuencia')
                    h, l = ax.get_legend_handles_labels()
                    if l:
                        ax.legend(fontsize=8)
                else:
                    ax.hist(x1, bins=20, range=(xmin, xmax), alpha=0.6, label='Sin error')
                    ax.hist(x2, bins=20, range=(xmin, xmax), alpha=0.6, label='Con error')
                    if log_y:
                        ax.set_yscale('log')
                    ax.set_xlabel('Valor')
                    ax.set_ylabel('Frecuencia')
                    h, l = ax.get_legend_handles_labels()
                    if l:
                        ax.legend(fontsize=8)

        elif tipo == "pie":
            counts_ok = data1.value_counts()
            counts_err = data2.value_counts()
            all_index = sorted(set(counts_ok.index) | set(counts_err.index))
            counts_ok = counts_ok.reindex(all_index, fill_value=0)
            counts_err = counts_err.reindex(all_index, fill_value=0)

            ok_plot = np.log1p(counts_ok.values) if use_log else counts_ok.values
            err_plot = np.log1p(counts_err.values) if use_log else counts_err.values

            if ok_plot.sum() == 0 and err_plot.sum() == 0:
                ax.text(0.5, 0.5, "Sin datos", ha='center', va='center')
            else:
                colors_ok = plt.cm.Blues(np.linspace(0.4, 0.9, len(all_index)))
                colors_err = plt.cm.Reds(np.linspace(0.4, 0.9, len(all_index)))

                # Externo = OK
                ax.pie(
                    ok_plot,
                    labels=[str(c) for c in all_index] if ok_plot.sum() > 0 else None,
                    autopct='%1.1f%%' if ok_plot.sum() > 0 else None,
                    startangle=90, counterclock=False,
                    pctdistance=0.8, radius=1.2,
                    wedgeprops=dict(width=0.35, edgecolor="w", alpha=0.9),
                    colors=colors_ok
                )
                # Interno = ERROR
                ax.pie(
                    err_plot,
                    labels=None,
                    autopct='%1.1f%%' if err_plot.sum() > 0 else None,
                    startangle=90, counterclock=False,
                    pctdistance=0.75, radius=0.8,
                    wedgeprops=dict(width=0.35, edgecolor="w", alpha=0.9),
                    colors=colors_err
                )

                # Leyendas
                legend_patches = [
                    Patch(facecolor=plt.cm.Blues(0.7), label='Sin error (anillo externo)'),
                    Patch(facecolor=plt.cm.Reds(0.7), label='Con error (anillo interno)')
                ]
                ax.legend(handles=legend_patches, loc='center left',
                          bbox_to_anchor=(1, 0.5), fontsize=8)
                ax.axis('equal')
                ax.set_title(param, fontweight='bold', fontsize=12, y=1.08)

        else:
            ax.text(0.5, 0.5, f"Tipo no soportado: {tipo}", ha='center', va='center')

        ax.set_title(param, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        h, l = ax.get_legend_handles_labels()
        if l:
            ax.legend(fontsize=8)

    for i in range(len(columnas_validas), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return parametros_constantes


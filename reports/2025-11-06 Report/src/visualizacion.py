import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

plt.style.use('default')
plt.rcParams['font.size'] = 10


def graficos_por_columna(df_sin_error=None, df_con_error=None, nombre_archivo=None,
                         tipo="histograma", log_y=False, use_log=True):
    """
    Gráficos comparando df_sin_error vs df_con_error.
    Si uno de los DataFrames es None, grafica solo el disponible.
    Tipos soportados: "histograma" y "pie".
    """

    if df_sin_error is not None and df_con_error is not None:
        columnas_parametros = sorted(set(df_sin_error.columns) & set(df_con_error.columns))
    elif df_sin_error is not None:
        columnas_parametros = sorted(df_sin_error.columns)
    else:
        columnas_parametros = sorted(df_con_error.columns)

    parametros_constantes = []
    columnas_validas = []

    for col in columnas_parametros:
        std_ok = pd.to_numeric(df_sin_error[col], errors='coerce').std() if df_sin_error is not None else np.nan
        std_err = pd.to_numeric(df_con_error[col], errors='coerce').std() if df_con_error is not None else np.nan

        if (std_ok == 0 or pd.isna(std_ok)) and (std_err == 0 or pd.isna(std_err)):
            parametros_constantes.append(col)
        else:
            columnas_validas.append(col)

    print(f"Creando gráficos tipo '{tipo}' para {len(columnas_validas)} parámetros NO constantes...")
    if parametros_constantes:
        print(f"Parámetros constantes (omitidos): {parametros_constantes}")

    if not columnas_validas:
        print("Todos los parámetros son constantes. No se generarán gráficos.")
        return parametros_constantes

    cols = 2 if tipo == "pie" else 3
    rows = (len(columnas_validas) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=120)

    titulo_archivo = f" - {nombre_archivo}" if nombre_archivo else ""
    fig.suptitle(
        f'Análisis de Parámetros{titulo_archivo}\n'
        f'Tipo de gráfico: {tipo} | {len(columnas_validas)} parámetros variables',
        fontsize=16, fontweight='bold', y=0.98
    )

    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, param in enumerate(columnas_validas):
        ax = axes_flat[i]
        data_ok = df_sin_error[param].dropna() if df_sin_error is not None else pd.Series(dtype=float)
        data_err = df_con_error[param].dropna() if df_con_error is not None else pd.Series(dtype=float)

        if tipo == "histograma":
            x_ok = pd.to_numeric(data_ok, errors="coerce").dropna().values
            x_err = pd.to_numeric(data_err, errors="coerce").dropna().values

            if df_sin_error is None:
                ax.hist(x_err, bins=20, alpha=0.7, color='red', label='Con error')
            elif df_con_error is None:
                ax.hist(x_ok, bins=20, alpha=0.7, color='blue', label='Sin error')
            else:
                ax.hist(x_ok, bins=20, alpha=0.5, label='Sin error', color='blue')
                ax.hist(x_err, bins=20, alpha=0.5, label='Con error', color='red')

            if log_y:
                ax.set_yscale('log')
            ax.set_xlabel(param)
            ax.set_ylabel("Frecuencia")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        elif tipo == "pie":
            counts_ok = data_ok.value_counts()
            counts_err = data_err.value_counts()
            all_index = sorted(set(counts_ok.index) | set(counts_err.index))
            counts_ok = counts_ok.reindex(all_index, fill_value=0)
            counts_err = counts_err.reindex(all_index, fill_value=0)

            ok_plot = np.log1p(counts_ok.values) if use_log else counts_ok.values
            err_plot = np.log1p(counts_err.values) if use_log else counts_err.values

            if df_sin_error is None:
                ax.pie(err_plot, labels=[str(c) for c in all_index],
                       autopct='%1.1f%%', startangle=90, colors=plt.cm.Reds(np.linspace(0.4, 0.9, len(all_index))))
                ax.set_title(f"{param} (solo errores)", fontsize=12)
            elif df_con_error is None:
                ax.pie(ok_plot, labels=[str(c) for c in all_index],
                       autopct='%1.1f%%', startangle=90, colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(all_index))))
                ax.set_title(f"{param} (solo sin error)", fontsize=12)
            else:
                ax.pie(ok_plot, radius=1.2, colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(all_index))),
                       startangle=90, counterclock=False, wedgeprops=dict(width=0.3, edgecolor="w"))
                ax.pie(err_plot, radius=0.8, colors=plt.cm.Reds(np.linspace(0.4, 0.9, len(all_index))),
                       startangle=90, counterclock=False, wedgeprops=dict(width=0.3, edgecolor="w"))
                ax.set_title(param, fontsize=12)
            ax.axis('equal')

        else:
            ax.text(0.5, 0.5, f"Tipo no soportado: {tipo}", ha='center', va='center')

    for i in range(len(columnas_validas), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return parametros_constantes



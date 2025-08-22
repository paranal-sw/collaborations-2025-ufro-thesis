import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

plt.style.use('default')
plt.rcParams['font.size'] = 10 

from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

def graficos_por_columna(df_sin_error, df_con_error, nombre_archivo=None, 
                         tipo="dispersion", log_y=False, use_log=True):
    """
    Graficos de distintos tipos que compara df_sin_error vs df_con_error.
    
    Parámetros:
        df_sin_error : DataFrame
        df_con_error : DataFrame
        nombre_archivo : str, opcional
        tipo : str ("dispersion", "histograma", "densidad", "pie")
        log_y : bool -> solo aplica a histograma
        use_log : bool -> solo aplica a pie
    """
    columnas_parametros = [col for col in df_sin_error.columns if col != 'param']
    parametros_constantes = []

    columnas_validas = []
    for col in columnas_parametros:
        std_ok = df_sin_error[col].std()
        std_err = df_con_error[col].std()
        if std_ok == 0 and std_err == 0:
            parametros_constantes.append(col)
        else:
            columnas_validas.append(col)

    print(f"Creando gráficos tipo '{tipo}' para {len(columnas_validas)} parámetros NO constantes...")
    print(f"Parámetros constantes (omitidos): {parametros_constantes}")

    n_params = len(columnas_validas)
    if n_params == 0:
        print("Todos los parámetros son constantes. No se generarán gráficos.")
        return parametros_constantes

    cols = 3 if tipo != "pie" else 2
    rows = (n_params + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), dpi=120)

    titulo_archivo = f" - {nombre_archivo}" if nombre_archivo else ""
    fig.suptitle(
        f'Análisis Comparativo de Parámetros{titulo_archivo}\n'
        f'Tipo de gráfico: {tipo} | {n_params} parámetros variables',
        fontsize=16, fontweight='bold', y=0.98
    )

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    for i, param in enumerate(columnas_validas):
        ax = axes_flat[i]

        data1 = df_sin_error[param].dropna()
        data2 = df_con_error[param].dropna()

        # DISPERSIÓN
        if tipo == "dispersion":
            ax.scatter(range(len(data1)), data1, c='royalblue', alpha=0.6, s=20,
                       label=f'Sin error (n={len(data1)})',
                       edgecolors='black', linewidth=0.3)
            ax.scatter(range(len(data2)), data2, c='tomato', alpha=0.6, s=20,
                       label=f'Con error (n={len(data2)})',
                       edgecolors='black', linewidth=0.3)
            ax.set_xlabel('Índice de Observación')
            ax.set_ylabel('Valor del Parámetro')

        # HISTOGRAMA
        elif tipo == "histograma":
            ax.hist(data1, bins=20, alpha=0.6, color='royalblue', label='Sin error')
            ax.hist(data2, bins=20, alpha=0.6, color='tomato', label='Con error')
            ax.set_xlabel('Valor del Parámetro')
            ax.set_ylabel('Frecuencia')
            if log_y:
                ax.set_yscale('log')

        # DENSIDAD
        elif tipo == "densidad":
            if len(data1) > 1 and np.std(data1) > 1e-8:
                kde1 = gaussian_kde(data1)
            else:
                kde1 = None
            if len(data2) > 1 and np.std(data2) > 1e-8:
                kde2 = gaussian_kde(data2)
            else:
                kde2 = None

            if (kde1 or kde2) and len(data1) > 0 and len(data2) > 0:
                x_min = min(data1.min(), data2.min())
                x_max = max(data1.max(), data2.max())
                if x_min != x_max:
                    x_vals = np.linspace(x_min, x_max, 200)
                else:
                    x_vals = None
            else:
                x_vals = None

            if kde1 and x_vals is not None:
                ax.plot(x_vals, kde1(x_vals), color='royalblue', lw=2, label='Sin error')
                ax.fill_between(x_vals, kde1(x_vals), color='royalblue', alpha=0.3)
            elif len(data1) > 0:
                ax.axvline(np.mean(data1), color='royalblue', linestyle='--', label='Sin error')

            if kde2 and x_vals is not None:
                ax.plot(x_vals, kde2(x_vals), color='tomato', lw=2, label='Con error')
                ax.fill_between(x_vals, kde2(x_vals), color='tomato', alpha=0.3)
            elif len(data2) > 0:
                ax.axvline(np.mean(data2), color='tomato', linestyle='--', label='Con error')

            ax.set_xlabel('Valor del Parámetro')
            ax.set_ylabel('Densidad')

        # PIE CHART
        elif tipo == "pie":
            counts_ok = data1.value_counts()
            counts_err = data2.value_counts()

            all_index = set(counts_ok.index) | set(counts_err.index)
            counts_ok = counts_ok.reindex(all_index, fill_value=0)
            counts_err = counts_err.reindex(all_index, fill_value=0)

            if use_log:
                counts_ok = np.log1p(counts_ok)
                counts_err = np.log1p(counts_err)

            colors_ok = plt.cm.Blues(np.linspace(0.4, 0.9, len(all_index)))   # tonos de azul
            colors_err = plt.cm.Reds(np.linspace(0.4, 0.9, len(all_index)))   # tonos de rojo

            # Pie externo = Con error
            wedges_err, _, autotexts_err = ax.pie(
                counts_err, labels=None, autopct='%1.1f%%',
                startangle=90, counterclock=False,
                pctdistance=0.75, radius=1.2,
                wedgeprops=dict(width=0.35, edgecolor="w", alpha=0.85),
                colors=colors_err
            )

            # Pie interno = Sin error
            wedges_ok, texts_ok, autotexts_ok = ax.pie(
                counts_ok, labels=counts_ok.index, autopct='%1.1f%%',
                startangle=90, counterclock=False,
                pctdistance=0.7, radius=0.8,
                wedgeprops=dict(width=0.35, edgecolor="w"),
                colors=colors_ok
            )

            for t in autotexts_err + autotexts_ok:
                t.set_fontsize(7)
                t.set_color("black")

            ax.legend(wedges_ok, counts_ok.index,
                title="Categorías del parámetro",
                loc='center left', bbox_to_anchor=(1, 0.5),
                fontsize=8)

            legend_patches = [
                Patch(facecolor=plt.cm.Blues(0.7), label='Sin error (anillo interno)'),
                Patch(facecolor=plt.cm.Reds(0.7), label='Con error (anillo externo)')
            ]
            fig.legend(handles=legend_patches,
               loc='lower center', bbox_to_anchor=(0.5, -0.02),
               title="Comparación de Anillos", ncol=2,
               fontsize=9, fancybox=True, shadow=True)

            ax.set_title(param, fontweight='bold', fontsize=12, y=1.1)
            ax.axis('equal')


        ax.set_title(param, fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(len(columnas_validas), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return parametros_constantes

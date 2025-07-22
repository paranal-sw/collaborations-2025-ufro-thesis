import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np 

plt.style.use('default')
plt.rcParams['font.size'] = 10 

def graficos_dispersion_por_columna(df_sin_error, df_con_error,nombre_archivo=None):
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

    print(f"Creando gráficos para {len(columnas_validas)} parámetros NO constantes...")
    print(f"Parámetros constantes (omitidos): {parametros_constantes}")
    print(f"Observaciones sin error: {len(df_sin_error)}")
    print(f"Observaciones con error: {len(df_con_error)}")

    n_params = len(columnas_validas)
    if n_params == 0:
        print("Todos los parámetros son constantes. No se generarán gráficos.")
        return parametros_constantes

    cols = 3
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    titulo_archivo = f" - {nombre_archivo}" if nombre_archivo else ""
    fig.suptitle(f'Análisis Comparativo de Parámetros por Condición de Error{titulo_archivo}\n' +
                 f'Distribución de {n_params} parámetros variables | ' +
                 f'Muestras: {len(df_sin_error)} sin error vs {len(df_con_error)} con error',                   
                 fontsize=16, fontweight='bold', y=0.98)     

    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    for i, param in enumerate(columnas_validas):
        ax = axes_flat[i]
        x_sin_error = range(len(df_sin_error))
        x_con_error = range(len(df_con_error))
        y_sin_error = df_sin_error[param]
        y_con_error = df_con_error[param]

        ax.scatter(x_sin_error, y_sin_error, c='blue', alpha=0.6, s=20,
                   label=f'Sin error (n={len(df_sin_error)})', edgecolors='darkred', linewidth=0.3)

        ax.scatter(x_con_error, y_con_error, c='red', alpha=0.6, s=20,
                   label=f'Con error (n={len(df_con_error)})', edgecolors='darkblue', linewidth=0.3)

        ax.set_title(param, fontweight='bold', fontsize=11)
        ax.set_xlabel('Índice de Observación')
        ax.set_ylabel('Valor del Parámetro')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for i in range(len(columnas_validas), len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    return parametros_constantes


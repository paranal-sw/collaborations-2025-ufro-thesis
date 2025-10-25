import pandas as pd
import os
import json

def codificar_categoricas_por_columna(df_categorico):
    """
    Codifica valores categóricos columna por columna, asignando enteros desde 1 en orden alfabético.
    
    Args:
        df_categorico (pd.DataFrame): DataFrame que contiene columnas categóricas.

    Returns:
        tuple: (pd.DataFrame codificado, dict de codificadores).
    """
    columnas_codificadas = {}
    codificadores = {}

    for col in df_categorico.columns:
        categorias = sorted(df_categorico[col].dropna().unique())
        mapa = {cat: i + 1 for i, cat in enumerate(categorias)}
        columnas_codificadas[col] = df_categorico[col].map(mapa)
        codificadores[col] = mapa

    df_codificado = pd.DataFrame(columnas_codificadas, index=df_categorico.index).copy()
    return df_codificado, codificadores

def procesar_strval_in_memory(df_dict, diccionario_json_path=None):
    """
    Procesa un diccionario de DataFrames con valores categóricos en memoria, codificándolos.
    Opcionalmente, guarda un diccionario de los mapeos de codificación en formato JSON.
    
    Args:
        df_dict (dict): Diccionario donde las claves son identificadores (ej. nombres de archivo)
                        y los valores son DataFrames con columnas categóricas.
        diccionario_json_path (str, optional): Ruta para guardar el diccionario de codificación
                                                en formato JSON. Si es None, no se guarda.

    Returns:
        tuple: (dict de DataFrames codificados, dict de codificadores globales).
    """
    diccionario_global = {}
    resultados_codificados = {}

    for name, df_categorico in df_dict.items():
        if df_categorico.empty:
            print(f"DataFrame '{name}' está vacío. Se omite la codificación.")
            continue

        df_codificado, mapas = codificar_categoricas_por_columna(df_categorico)
        diccionario_global[name] = mapas
        resultados_codificados[name] = df_codificado
    
    if diccionario_json_path:
        # Asegurarse de que el directorio exista si se va a guardar el JSON
        output_dir = os.path.dirname(diccionario_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        with open(diccionario_json_path, "w", encoding='utf-8') as f:
            json.dump(diccionario_global, f, ensure_ascii=False, indent=2)
        print(f"Diccionario de codificación guardado en: {diccionario_json_path}")
    
    return resultados_codificados, diccionario_global
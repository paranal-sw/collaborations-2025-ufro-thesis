import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def extract_params(df_original):
    df = df_original.copy()
    df = df[df['@timestamp'] < df['@timestamp'].min() + pd.Timedelta(seconds=3)]
    df = df[df['logtext'].str.contains(r"^.* = .*$")]
    df['param'] = df['logtext'].str.split(' = ').str[0].str.replace(" ", ".")
    df['value'] = df['logtext'].str.split(' = ').str[1].str.replace("'", "")
    df['value'] = df['value'].str.replace(' K', '', regex=False)
    df['numval'] = pd.to_numeric(df['value'], errors='coerce')
    df['strval'] = df['value'].where(df['numval'].isna(), "")
    df = df.drop(columns=['value'])
    df = df.set_index('param').sort_index()
    return df

def extract_trace_parameters(df_meta, df_traces, extract_params, tpl_id):
    tpl_indices = df_meta[df_meta['TPL_ID'] == tpl_id].index
    df_filtered = df_traces[df_traces['trace_id'].isin(tpl_indices)]
    params_dict = {
        trace_id: extract_params(df_filtered[df_filtered['trace_id'] == trace_id])
        .loc[lambda df: ~df.index.duplicated(keep='first')]
        for trace_id in df_filtered['trace_id'].unique()
    }
    params_df = pd.concat(
        {trace_id: df['numval'] for trace_id, df in params_dict.items()},
        axis=1).T
    params_df = params_df.sort_index(axis=1).dropna(how='all').dropna(axis=1, how='all')
    return params_df

def extract_trace_categoricals(df_meta, df_traces, extract_params, tpl_id='MATISSE_img_acq'):
    tpl_indices = df_meta[df_meta['TPL_ID'] == tpl_id].index
    df_filtered = df_traces[df_traces['trace_id'].isin(tpl_indices)]
    params_dict = {
        trace_id: extract_params(df_filtered[df_filtered['trace_id'] == trace_id])
        .loc[lambda df: ~df.index.duplicated(keep='first')]
        for trace_id in df_filtered['trace_id'].unique()
    }
    strval_df = pd.concat(
        {trace_id: df['strval'] for trace_id, df in params_dict.items()},
        axis=1).T
    strval_df = strval_df.sort_index(axis=1).dropna(how='all').dropna(axis=1, how='all')
    return strval_df

def min_max_normalizar(df, min_val=0, max_val=1):
    scaler = MinMaxScaler(feature_range=(min_val, max_val))
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

def cargar_y_procesar_carpeta(input_dir, prefix, tpl_id_filter=None, threshold=0.5):
    import os
    resultados_procesados = {}
    for filename in os.listdir(input_dir):
        if filename.startswith(prefix) and filename.endswith(".csv"):
            tpl_id = filename.replace(prefix, "").replace(".csv", "")
            if tpl_id_filter and tpl_id not in tpl_id_filter:
                continue
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path, index_col=0)
            df = df.loc[:, df.isna().mean() < threshold]
            df = df.loc[df.isna().mean(axis=1) < threshold, :]
            if not df.empty:
                df_normalizado = min_max_normalizar(df)
                if not df_normalizado.empty:
                    resultados_procesados[filename] = df_normalizado
    return resultados_procesados

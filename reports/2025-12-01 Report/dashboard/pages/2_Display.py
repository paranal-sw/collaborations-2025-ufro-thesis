import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]       # -> carpeta "2025-12-01 Report"
ARTE_DIR = ROOT / "dashboard" / "artefactos"
FIG_DIR = ROOT / "figuras"


st.markdown("""
<style>
html, body, [class*="css"] {
    font-size: 16px;
}
.section-title {
    font-size: 24px;
    font-weight: 800;
    color: #1f4172;
    margin-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">2 — Display por Instrumento y TPL_ID</div>',
            unsafe_allow_html=True)

# Timestamp de la última generación
ts_path = ARTE_DIR / "timestamps.json"
if ts_path.exists():
    with ts_path.open() as f:
        ts = json.load(f).get("last_run", "desconocido")
    st.info(f"Última generación de artefactos: {ts}")
else:
    st.warning("Aún no se encuentra timestamps.json. Ejecuta primero la página 1 (Train).")

# Selector de instrumento
instrumentos = {
    "matisse": "artefacto_matisse.parquet",
    "gravity": "artefacto_gravity.parquet",
    "pionier": "artefacto_pionier.parquet",
}

instrumento = st.selectbox("Instrumento:", list(instrumentos.keys()))

artefacto_path = ARTE_DIR / instrumentos[instrumento]

if not artefacto_path.exists():
    st.error(f"No se encontró el artefacto: {artefacto_path}")
    st.stop()

# Cargar artefacto y seleccionar TPL_ID
df = pd.read_parquet(artefacto_path)

if "TPL_ID" not in df.columns:
    st.error("El artefacto no contiene la columna 'TPL_ID'.")
    st.stop()

tpl_ids = sorted(df["TPL_ID"].dropna().unique())
tpl_id = st.selectbox("TPL_ID:", tpl_ids)

df_tpl = df[df["TPL_ID"] == tpl_id].copy()

st.write(f"Número de observaciones para **{tpl_id}**: {len(df_tpl)}")

# Tabs principales
tab_df, tab_figs, tab_clusters = st.tabs(
    ["DataFrame filtrado", "Figuras (numéricas / categóricas)", "Clusters"]
)

def mostrar_figura_png(tpl_id: str, sufijo: str, titulo: str):
    """
    Busca una imagen en FIG_DIR con nombre f'{tpl_id}_{sufijo}.png'
    sufijo: 'cluster', 'hist', 'pie'
    """
    filename = f"{tpl_id}_{sufijo}.png"
    ruta = FIG_DIR / filename
    if ruta.exists():
        st.markdown(f"**{titulo}**")
        st.image(str(ruta))
    else:
        st.info(f"No se encontró la figura: {filename}")

# Tab 1: DataFrame filtrado
with tab_df:
    st.subheader("DataFrame filtrado por TPL_ID (columnas no vacías)")

    # eliminar columnas completamente vacías para este TPL_ID
    df_tpl_trim = df_tpl.dropna(axis=1, how="all")

    st.dataframe(df_tpl_trim, use_container_width=True)

# Tab 2: Figuras numéricas / categóricas 
with tab_figs:
    st.subheader("Figuras generadas desde los notebooks")

    col1, col2 = st.columns(2)

    with col1:
        mostrar_figura_png(tpl_id, "hist", "Histograma de parámetros numéricos")

    with col2:
        mostrar_figura_png(tpl_id, "pie", "Pie de parámetros categóricos")

# Tab 3: Clusters
import altair as alt


with tab_clusters:
    st.subheader("Clusters: embedding y distribución")

    if "cluster" not in df_tpl.columns:
        st.info("El artefacto no contiene la columna 'cluster'.")
    else:
        # Conteo y porcentajes por cluster
        conteo = (
            df_tpl["cluster"]
            .value_counts()
            .sort_index()
        )
        df_clusters = conteo.rename("n_obs").reset_index()
        df_clusters.rename(columns={"cluster": "cluster"}, inplace=True)
        df_clusters["porcentaje"] = df_clusters["n_obs"] / df_clusters["n_obs"].sum() * 100

        col1, col2 = st.columns([2, 1])

        # izquierda: scatter de clusters (PNG)
        with col1:
            mostrar_figura_png(tpl_id, "cluster", "Scatter de clusters")

        # derecha: pie + tabla coloreada 
        with col2:
            st.markdown("**Distribución de observaciones por cluster**")

            # Pie de porcentajes
            chart = (
                alt.Chart(df_clusters)
                .mark_arc()
                .encode(
                    theta="n_obs:Q",
                    color="cluster:N",
                    tooltip=[
                        "cluster:N",
                        "n_obs:Q",
                        alt.Tooltip("porcentaje:Q", format=".1f", title="porcentaje (%)"),
                    ],
                )
            )
            st.altair_chart(chart, use_container_width=True)

            # Tabla con colores
            st.markdown("**Tabla de clusters**")
            df_show = df_clusters.copy()
            df_show["porcentaje"] = df_show["porcentaje"].map(lambda x: f"{x:.1f} %")

            st.table(
                df_show.style.background_gradient(
                    subset=["n_obs"], cmap="Reds"
                )
            )

import os
import streamlit as st


CARPETA_FIG = "figuras"

st.set_page_config(
    page_title="Dashboard TPL_ID – MATISSE / GRAVITY / PIONIER",
    layout="wide",
)


def detectar_tpl_con_figuras():
    """
    Detecta automáticamente qué TPL_ID tienen al menos una figura PNG.
    Retorna un diccionario:
        { "MATISSE": [...], "GRAVITY": [...], "PIONIER": [...] }
    """
    estructura = {"MATISSE": [], "GRAVITY": [], "PIONIER": []}

    if not os.path.isdir(CARPETA_FIG):
        return estructura

    archivos = os.listdir(CARPETA_FIG)
    tpl_set = set()

    for fname in archivos:
        base, ext = os.path.splitext(fname)
        if ext.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        # Detecta tpl quitando sufijos de tipo de gráfico
        for suf in ("_hist", "_pie", "_cluster"):
            if base.endswith(suf):
                tpl = base[: -len(suf)]
                tpl_set.add(tpl)
                break

    # Clasificación por instrumento
    for tpl in sorted(tpl_set):
        if tpl.startswith("MATISSE_") or tpl == "errseverity":
            estructura["MATISSE"].append(tpl)
        elif tpl.startswith("GRAVITY_") or tpl in {"Dual", "Fringe"}:
            estructura["GRAVITY"].append(tpl)
        elif tpl.startswith("PIONIER_"):
            estructura["PIONIER"].append(tpl)

    return estructura


def ruta_figura(tpl_id: str, tipo: str) -> str:
    return os.path.join(CARPETA_FIG, f"{tpl_id}_{tipo}.png")


def mostrar_imagen(tpl_id: str, tipo: str, titulo: str, texto_ayuda: str):
    st.subheader(titulo)
    st.caption(texto_ayuda)

    ruta = ruta_figura(tpl_id, tipo)

    if os.path.exists(ruta):
        st.image(ruta, use_container_width=True)
        st.caption(f"Figura: {os.path.basename(ruta)}")
    else:
        st.warning(f"No se encontró la figura: {os.path.basename(ruta)}")


estructura_tpl = detectar_tpl_con_figuras()
instrumentos_disponibles = [inst for inst, lista in estructura_tpl.items() if lista]

if not instrumentos_disponibles:
    st.error("No se encontraron imágenes en la carpeta 'figuras/'.")
    st.stop()

st.sidebar.title("Configuración")

instrumento = st.sidebar.selectbox(
    "Selecciona instrumento:",
    instrumentos_disponibles
)

tpl_lista = estructura_tpl[instrumento]

tpl_id = st.sidebar.selectbox(
    "Selecciona TPL_ID:",
    tpl_lista
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
Notas importantes:
- Solo se muestran TPL_ID que tienen figuras PNG disponibles.
- Las figuras pueden insertarse directamente en LaTeX o PowerPoint.
"""
)

if st.sidebar.button("Recargar lista de TPL_ID"):
    st.rerun()


st.title("Dashboard de resultados por TPL_ID")

st.markdown(
    """
Este panel resume los resultados generados en los notebooks:
- Histogramas de parámetros numéricos.
- Gráficos Pie para parámetros categóricos.
- Clustering con DBSCAN sobre parámetros normalizados (visualización PCA 2D).
"""
)

st.markdown("---")


c1, c2, c3 = st.columns([1.3, 1.5, 1.5])

with c1:
    st.markdown("#### Instrumento seleccionado")
    st.markdown(f"**{instrumento}**")

with c2:
    st.markdown("#### TPL_ID seleccionado")
    st.markdown(f"**{tpl_id}**")


st.markdown("---")


tabs = st.tabs(["Histogramas numéricos", "Pie Charts", "Clustering"])

with tabs[0]:
    mostrar_imagen(
        tpl_id,
        "hist",
        "Histogramas numéricos",
        "Distribución de parámetros numéricos obtenidos del dataset de errores."
    )

with tabs[1]:
    mostrar_imagen(
        tpl_id,
        "pie",
        "Parámetros categóricos – Pie Charts",
        "Resumen de los valores categóricos más frecuentes dentro de las observaciones con error."
    )

with tabs[2]:
    mostrar_imagen(
        tpl_id,
        "cluster",
        "Clustering (DBSCAN + PCA 2D + jitter)",
        "Visualización en 2D del clustering basado en parámetros normalizados."
    )




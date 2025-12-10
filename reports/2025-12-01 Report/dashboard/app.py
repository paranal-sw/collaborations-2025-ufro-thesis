from pathlib import Path
import os
import json
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ARTE_DIR = ROOT / "dashboard" / "artefactos"
FIG_DIR = ROOT / "figuras"

st.set_page_config(
    page_title="Dashboard Instrumentos ESO",
    layout="wide"
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-size: 16px;
    }
    .main-title {
        font-size: 30px;
        font-weight: 800;
        color: #1f4172;
        margin-bottom: 0.4rem;
    }
    .subtitle {
        font-size: 18px;
        color: #4a4a4a;
        margin-bottom: 1.2rem;
    }
    .info-box {
        padding: 1rem 1.4rem;
        border-radius: 0.9rem;
        background-color: #f5f7fb;
        border: 1px solid #dde3f0;
        margin-bottom: 0.8rem;
    }
    .info-title {
        font-weight: 700;
        color: #243b6b;
        margin-bottom: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">Dashboard de Instrumentos — MATISSE / GRAVITY / PIONIER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Exploración de artefactos de clustering por instrumento y TPL_ID.</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown('<div class="info-title">Páginas</div>', unsafe_allow_html=True)
st.markdown(
    """
    - **1 — Train**: ejecuta los notebooks de cada instrumento y regenera los artefactos (`.parquet`).  
    - **2 — Display**: selecciona instrumento y TPL_ID, muestra histogramas de parámetros numéricos, pies de parámetros categóricos, scatter por cluster, resumen de clusters y el DataFrame filtrado.
    """
)
st.markdown('</div>', unsafe_allow_html=True)

st.info("Usa la barra lateral para cambiar entre páginas.")

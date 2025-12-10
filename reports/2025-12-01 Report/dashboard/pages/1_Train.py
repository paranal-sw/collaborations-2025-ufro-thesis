import streamlit as st
import subprocess
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

st.set_page_config(layout="wide")

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

st.markdown('<div class="section-title">1 — Train: ejecutar notebooks y generar artefactos</div>', unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parents[2]

ARTE_DIR = ROOT / "dashboard" / "artefactos"
ARTE_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp anterior
ts_path = ARTE_DIR / "timestamps.json"
if ts_path.exists():
    with ts_path.open() as f:
        ts = json.load(f).get("last_run", "desconocido")
    st.info(f"Última generación previa: {ts}")
else:
    st.warning("Aún no hay timestamp previo.")

# Notebooks a ejecutar
NOTEBOOKS = {
    "matisse": "2025-12-01 Matisse.ipynb",
    "gravity": "2025-12-01 Gravity.ipynb",
    "pionier": "2025-12-01 Pionier.ipynb",
}

# Formato tiempo min + seg
def fmt(t):
    minutos = int(t // 60)
    segundos = int(t % 60)
    if minutos > 0:
        return f"{minutos} min {segundos} s"
    else:
        return f"{segundos} s"

# Ejecutar notebook (sin logs en vivo, pero con tiempo)
def run_notebook(nb_rel_path, name, status_box, timer_placeholder, log_container):
    """
    nb_rel_path: ruta relativa a ROOT, por ejemplo '2025-12-01 Matisse.ipynb'
    Ejecuta nbconvert con cwd=ROOT.
    Devuelve tiempo total en segundos.
    """
    start = time.time()

    # Informar inicio
    timer_placeholder.write("Tiempo transcurrido: 0 s")

    try:
        process = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook", "--execute", "--inplace", nb_rel_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ROOT),
        )
    except Exception as e:
        elapsed_total = time.time() - start
        timer_placeholder.write(f"Tiempo transcurrido: {fmt(elapsed_total)}")
        status_box.error(f"{name.upper()} lanzó una excepción al ejecutar nbconvert.")
        log_container.error(str(e))
        return elapsed_total

    elapsed_total = time.time() - start
    timer_placeholder.write(f"Tiempo transcurrido: {fmt(elapsed_total)}")

    stdout = process.stdout or ""
    stderr = process.stderr or ""
    log_text = ""
    if stdout.strip():
        log_text += stdout
    if stderr.strip():
        log_text += "\n[STDERR]\n" + stderr

    if log_text.strip():
        log_container.code(log_text)
    else:
        log_container.info("No hubo salida de nbconvert (stdout/stderr vacíos).")

    if process.returncode == 0:
        status_box.success(f"{name.upper()} completado en {fmt(elapsed_total)}")
    else:
        status_box.error(f"{name.upper()} terminó con errores (código {process.returncode})")

    return elapsed_total

if st.button("Ejecutar notebooks y generar artefactos"):

    st.info("Iniciando ejecución…")

    global_start = time.time()
    status_box = st.empty()
    timer_placeholder = st.empty()
    expander = st.expander("Ver logs completos", expanded=True)
    progress = st.progress(0)

    total_times = {}
    n_total = len(NOTEBOOKS)

    try:
        for i, (name, nb_rel_path) in enumerate(NOTEBOOKS.items(), start=1):
            status_box.markdown(f"### Ejecutando {name.upper()} ({i}/{n_total})…")
            with expander:
                st.subheader(f"Logs de {name.upper()}")
                log_container = st.empty()

            elapsed = run_notebook(nb_rel_path, name, status_box, timer_placeholder, log_container)
            total_times[name] = elapsed
            progress.progress(i / n_total)

    except Exception as e:
        st.error("Ocurrió un error inesperado en la ejecución de los notebooks.")
        st.exception(e)
    else:
        total_elapsed = time.time() - global_start
        st.success(f"Proceso completo en **{fmt(total_elapsed)}**")

        st.write("Detalle por instrumento:")
        for name in NOTEBOOKS.keys():
            t = total_times.get(name)
            if t is not None:
                st.write(f"- {name.upper()}: {fmt(t)}")

        # Guardar timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with ts_path.open("w") as f:
            json.dump({"last_run": ts}, f)
        st.info(f"Nuevo timestamp guardado: {ts}")

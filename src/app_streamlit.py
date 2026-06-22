"""
Application Streamlit de prédiction de la consommation électrique journalière.

Charge le modèle entraîné (output/modele_v2_optimise.joblib), propose un
formulaire (date + consommation passée + prévision météo), reconstruit les
variables attendues par le modèle et affiche la consommation prévue.

Lancer :
    pip install streamlit
    streamlit run app_streamlit.py
"""

import os
import datetime
import numpy as np
import pandas as pd
import joblib
import holidays
import streamlit as st

# ------------------------------------------------------------
# Chargement du modèle (mis en cache : chargé une seule fois)
# ------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "output", "model_prod.joblib")


@st.cache_resource
def charger_modele():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


if not os.path.exists(MODEL_PATH):
    st.error(f"Modèle introuvable : {MODEL_PATH}\n\n"
             "Lance d'abord hyperparameter_tuning.py pour générer le .joblib.")
    st.stop()

model, FEATURES = charger_modele()
fr_holidays = holidays.France()
JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# ------------------------------------------------------------
# Interface
# ------------------------------------------------------------
st.set_page_config(page_title="Prédiction consommation électrique", page_icon="⚡")
st.title("⚡ Prédiction de la consommation électrique journalière")
st.write("Renseigne un jour à prédire, la consommation passée et la prévision météo.")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input("Date à prédire", datetime.date(2026, 1, 15))
    conso_j1 = st.number_input("Consommation de la veille (MW)", value=62000.0, step=500.0)
    conso_j7 = st.number_input("Consommation il y a 7 jours (MW)", value=61000.0, step=500.0)
with col2:
    conso_moy_7j = st.number_input("Moyenne des 7 derniers jours (MW)", value=60500.0, step=500.0)
    temperature = st.number_input("Température moyenne prévue (°C)", value=4.5, step=0.5)
    temperature_max = st.number_input("Température maximale prévue (°C)", value=8.0, step=0.5)

# ------------------------------------------------------------
# Prédiction
# ------------------------------------------------------------
if st.button("Prédire la consommation", type="primary"):
    jsem = date.weekday()  # lundi = 0 ... dimanche = 6
    est_weekend = 1 if jsem >= 5 else 0
    est_ferie = 1 if date in fr_holidays else 0

    # mêmes calculs que dans prepare_dataset.py
    valeurs = {
        "Conso_J1":          conso_j1,
        "Conso_J7":          conso_j7,
        "Conso_moy_7j":      conso_moy_7j,
        "Temperature":       temperature,
        "Temperature_max":   temperature_max,
        "Jour_semaine_sin":  np.sin(2 * np.pi * jsem / 7),
        "Jour_semaine_cos":  np.cos(2 * np.pi * jsem / 7),
        "Est_weekend":       est_weekend,
        "Est_ferie":         est_ferie,
    }
    X = pd.DataFrame([[valeurs[f] for f in FEATURES]], columns=FEATURES)

    prevision = float(model.predict(X)[0])

    st.success(f"### Consommation prévue : {prevision:,.0f} MW".replace(",", " "))
    st.caption(
        f"{JOURS[jsem]} {date:%d/%m/%Y}  —  "
        f"week-end : {'oui' if est_weekend else 'non'}  —  "
        f"férié : {'oui' if est_ferie else 'non'}"
    )
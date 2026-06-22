"""
Le test de monitoring post-déploiement a pour objectif de vérifier que le modèle
actuellement en production sur Render performe toujours correctement sur les données
de test réelles. Il envoie les données du jeu de test à l'API déployée, collecte les
prédictions retournées et calcule les métriques de performance (R², MAPE). Ces métriques
sont ensuite comparées aux références enregistrées dans le CSV de production afin de
détecter un éventuel drift ou une dégradation des performances en prod. Ce test est
destiné à être lancé après chaque déploiement ou périodiquement pour s'assurer de la
stabilité du modèle en conditions réelles.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ============================================================
# URL DE L'API RENDER
# ============================================================

BASE_URL = "https://api-prediction-consomation-energie.onrender.com"

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT_DIR, "src", "output", "dataset_journalier_enrichi.csv")
CSV_PROD  = os.path.join(ROOT_DIR, "src", "output", "resultats_optimisation.csv")

REPORT_DIR = os.path.join(ROOT_DIR, "test", "resultat_test")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(REPORT_DIR, "rapport_monitoring_prod.txt")

ANNEE_TEST = 2023

# ============================================================
def write(line: str):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ============================================================
# MONITORING POST-DÉPLOIEMENT
# ============================================================

def test_monitoring_prod():

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write("\n" + "=" * 60)
    write(f"RUN DATE : {run_time}")
    write(f"API cible : {BASE_URL}")
    write("=" * 60 + "\n")

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    assert "Date" in df.columns, "Colonne 'Date' manquante dans le dataset."

    TARGET = "Consommation"
    test = df[df["Annee"] >= ANNEE_TEST].dropna(subset=[TARGET])
    test = test.sort_values("Date").head(100)

    write(f"Période test  : Annee >= {ANNEE_TEST}")
    write(f"Jours de test : {len(test)}\n")

    # ----------------------------------------------------------
    # BASELINE CSV
    # ----------------------------------------------------------
    assert os.path.exists(CSV_PROD)

    csv = pd.read_csv(CSV_PROD)
    xgb = csv[csv["Modèle"] == "XGBoost"].iloc[0]

    ref_r2 = float(xgb["R2_test"])
    ref_mape = float(xgb["MAPE_test_%"])

    write("--- BASELINE ENREGISTRÉE (modèle prod) ---")
    write(f"R2 ref   : {ref_r2:.4f}")
    write(f"MAPE ref : {ref_mape:.4f} %\n")

    # ----------------------------------------------------------
    # APPELS API RENDER
    # ----------------------------------------------------------
    predictions = []
    y_true = []
    echecs = 0

    write("--- APPELS API RENDER ---")

    for _, row in test.iterrows():

        # ⚠️ IMPORTANT : payload BRUT uniquement (pas de features dérivées)
        payload = {
            "date": str(row["Date"]),
            "conso_j1": float(row["Conso_J1"]),
            "conso_j7": float(row["Conso_J7"]),
            "conso_moy_7j": float(row["Conso_moy_7j"]),
            "temperature": float(row["Temperature"]),
            "temperature_max": float(row["Temperature_max"])
        }

        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                pred = response.json().get("consommation_prevue_MW")
                if pred is not None:
                    predictions.append(pred)
                    y_true.append(float(row[TARGET]))
                else:
                    echecs += 1
            else:
                echecs += 1

        except Exception:
            echecs += 1

    write(f"Requêtes envoyées   : {len(test)}")
    write(f"Prédictions valides : {len(predictions)}")
    write(f"Échecs / erreurs    : {echecs}\n")

    assert len(predictions) > 0, "Aucune prédiction valide retournée par l'API Render."

    # ----------------------------------------------------------
    # MÉTRIQUES
    # ----------------------------------------------------------
    r2_prod = r2_score(y_true, predictions)
    mape_prod = mean_absolute_percentage_error(y_true, predictions) * 100

    r2_drop = ((ref_r2 - r2_prod) / ref_r2) * 100 if ref_r2 != 0 else float("inf")
    mape_diff = mape_prod - ref_mape

    write("--- PERFORMANCE API RENDER (prod) ---")
    write(f"R2 prod   : {r2_prod:.4f}")
    write(f"MAPE prod : {mape_prod:.4f} %\n")

    write("--- ÉCARTS vs BASELINE ---")
    write(f"R2 drop (%)             : {r2_drop:.2f}  (seuil : > 5 %)")
    write(f"MAPE variation (points) : {mape_diff:.2f}  (seuil : > 2 pts)\n")

    # ----------------------------------------------------------
    # VERDICT
    # ----------------------------------------------------------
    if r2_drop > 5 or abs(mape_diff) > 2:
        verdict = "⚠️ DRIFT DÉTECTÉ — performances dégradées"
        result = 1
    else:
        verdict = "✅ PROD STABLE"
        result = 0

    write("--- VERDICT ---")
    write(verdict)
    write("=" * 60 + "\n")

    return result
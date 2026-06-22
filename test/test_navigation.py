"""
Les tests automatisés ont pour objectif de valider le bon fonctionnement de l’API de
prédiction de consommation électrique, sa robustesse et sa performance avant mise en
production. Ils vérifient que la page d’accueil est accessible et contient bien 
l’interface de démonstration, que le endpoint /health confirme le bon état du service
et le chargement du modèle, et que le endpoint /predict retourne des prédictions cohérentes
à partir de données valides. Les tests contrôlent également la gestion des erreurs, notamment 
le rejet des formats de date invalides avec un code HTTP 400, afin de garantir la robustesse du 
système face aux entrées incorrectes. Enfin, ils mesurent le temps de réponse des requêtes afin 
d’évaluer la performance globale de l’API. L’ensemble des tests est exécuté automatiquement avec 
pytest et FastAPI TestClient, simulant des requêtes HTTP et validant les réponses via des assertions. 
Les résultats sont enregistrés dans un fichier de rapport afin de conserver un historique des 
exécutions et de suivre l’évolution de la qualité du système dans le temps.

"""


import sys
import os
import time
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ============================================================
# Dossier des résultats
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "resultat_test")
os.makedirs(RESULT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(RESULT_DIR, "resultat_tests_navigation.txt")


# ============================================================
# LOG FUNCTION
# ============================================================
def log_result(f, test_name, status, details=""):
    f.write(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        f"{test_name} : {status}"
    )
    if details:
        f.write(f" - {details}")
    f.write("\n")


# ============================================================
# INIT NEW TEST RUN BLOCK
# ============================================================
def setup_module(module):

    with open(REPORT_FILE, "a", encoding="utf-8") as f:

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"NOUVEAU RAPPORT DE TEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")



# ============================================================
# TEST HOME
# ============================================================
def test_home():
    start = time.time()
    response = client.get("/")
    duration = round(time.time() - start, 4)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        try:
            assert response.status_code == 200
            assert "Prédiction de consommation électrique" in response.text

            log_result(f, "test_home", "SUCCESS", f"Temps={duration}s")

        except AssertionError as e:
            log_result(f, "test_home", "FAILED", f"{e} | Temps={duration}s")
            raise


# ============================================================
# TEST HEALTH
# ============================================================
def test_health():
    start = time.time()
    response = client.get("/health")
    duration = round(time.time() - start, 4)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        try:
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["modele_charge"] is True

            log_result(f, "test_health", "SUCCESS", f"Temps={duration}s")

        except AssertionError as e:
            log_result(f, "test_health", "FAILED", f"{e} | Temps={duration}s")
            raise


# ============================================================
# TEST PREDICT
# ============================================================
def test_predict():
    payload = {
        "date": "2026-01-15",
        "conso_j1": 62000,
        "conso_j7": 61000,
        "conso_moy_7j": 60500,
        "temperature": 4.5,
        "temperature_max": 8.0
    }

    start = time.time()
    response = client.post("/predict", json=payload)
    duration = round(time.time() - start, 4)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        try:
            assert response.status_code == 200

            data = response.json()
            assert "consommation_prevue_MW" in data
            assert data["date"] == "2026-01-15"

            log_result(
                f,
                "test_predict",
                "SUCCESS",
                f"Pred={data['consommation_prevue_MW']} MW | Temps={duration}s"
            )

        except AssertionError as e:
            log_result(f, "test_predict", "FAILED", f"{e} | Temps={duration}s")
            raise


# ============================================================
# TEST INVALID DATE
# ============================================================
def test_predict_invalid_date():
    payload = {
        "date": "15/01/2026",
        "conso_j1": 62000,
        "conso_j7": 61000,
        "conso_moy_7j": 60500,
        "temperature": 4.5,
        "temperature_max": 8.0
    }

    start = time.time()
    response = client.post("/predict", json=payload)
    duration = round(time.time() - start, 4)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        try:
            assert response.status_code == 400
            log_result(f, "test_predict_invalid_date", "SUCCESS", f"Temps={duration}s")

        except AssertionError as e:
            log_result(f, "test_predict_invalid_date", "FAILED", f"{e} | Temps={duration}s")
            raise
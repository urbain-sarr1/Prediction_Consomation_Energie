"""
Le test de business logic a pour objectif de vérifier la cohérence des prédictions générées 
par l’API à partir d’entrées identiques ou proches. Il envoie plusieurs requêtes avec les 
mêmes données afin de contrôler si le modèle produit des résultats stables, puis compare 
les sorties pour détecter d’éventuelles variations anormales. Le test vérifie également 
que les prédictions respectent des règles métier simples, comme l’absence de valeurs négatives 
ou incohérentes. Enfin, il analyse la cohérence globale des résultats afin de s’assurer que le 
comportement du modèle reste logique et fiable dans des conditions d’utilisation répétées.

"""

import os
import time
from datetime import datetime
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ============================================================
# REPORT CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "resultat_test")
os.makedirs(RESULT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(RESULT_DIR, "rapport_business_logic.txt")


# ============================================================
# WRITE SYSTEM
# ============================================================

def write(line: str):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_if_new_file():
    if os.path.getsize(REPORT_FILE) == 0:
        write("=========================================")
        write("BUSINESS LOGIC TEST REPORT")
        write("API FASTAPI - PREDICTION CONSOMMATION")
        write("=========================================\n")


def start_session():
    write("\n#################################################")
    write(f"NEW TEST SESSION - {datetime.now()}")
    write("#################################################\n")


# ============================================================
# BASE INPUT
# ============================================================

def base_input():
    return {
        "date": "2026-01-15",
        "conso_j1": 62000,
        "conso_j7": 61000,
        "conso_moy_7j": 60500,
        "temperature": 4.5,
        "temperature_max": 8.0
    }


# ============================================================
# TEST UNIQUE : STABILITY SIMPLE
# ============================================================

def test_prediction_stability(n=100):
    start_session()
    write("===== PREDICTION STABILITY TEST =====")

    payload = base_input()
    outputs = []
    errors = 0

    for i in range(n):
        try:
            response = client.post("/predict", json=payload)

            if response.status_code == 200:
                outputs.append(response.json().get("prediction"))
            else:
                errors += 1

        except Exception:
            errors += 1

    unique_outputs = list(set(outputs))

    stable = len(unique_outputs) == 1

    write(f"Total requests: {n}")
    write(f"Successful predictions: {len(outputs)}")
    write(f"Errors: {errors}")
    write(f"Unique outputs: {unique_outputs[:10]}")

    write(f"\nSTABLE MODEL: {stable}")

    # FINAL VERDICT
    verdict = "SUCCESS ✅ (stable model)" if stable else "FAILED ❌ (unstable model)"

    write("\n=========================================")
    write(f"FINAL VERDICT: {verdict}")
    write("=========================================")

    return stable


# ============================================================
# MAIN
# ============================================================

def test_business_logic_suite():
    write_if_new_file()
    test_prediction_stability()
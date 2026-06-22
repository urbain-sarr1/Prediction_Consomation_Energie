"""
L'ensemble des tests réalisés (chaos/fuzz, performance et stress) permet d'évaluer 
de manière complète le comportement de l'API de prédiction de consommation dans différentes 
conditions d'utilisation. Le test chaos/fuzz consiste à envoyer des données aléatoires, 
parfois invalides, afin de vérifier la robustesse du système et sa capacité à ne pas planter 
face à des entrées imprévues. Le test de performance mesure quant à lui les temps de réponse 
de l'API avec des données valides, afin d'évaluer sa rapidité et sa stabilité à travers des 
indicateurs comme la latence moyenne ainsi que les percentiles P95 et P99. Enfin, le test de 
stress simule une forte charge de requêtes afin d'analyser la capacité de l'API à maintenir 
ses performances sous pression et à traiter un nombre élevé de requêtes par seconde. 
Dans l'ensemble, ces tests permettent de confirmer la stabilité, la rapidité et la 
robustesse du service dans des conditions normales et extrêmes d'utilisation.
"""

import random
import time
import statistics
import os
import requests
from datetime import datetime

# ============================================================
# URL DE L'API RENDER
# ============================================================

BASE_URL = "https://api-prediction-consomation-energie.onrender.com"

# ============================================================
# REPORT CONFIG
# ============================================================

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "resultat_test")
os.makedirs(RESULT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(RESULT_DIR, "rapport_chaos_tests.txt")


# ============================================================
# WRITE SYSTEM
# ============================================================

def write(line: str):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_if_new_file():
    if os.path.getsize(REPORT_FILE) == 0:
        write("=========================================")
        write("CHAOS / FUZZ / PERFORMANCE TEST REPORT")
        write("API FASTAPI - PREDICTION CONSOMMATION")
        write(f"API cible : {BASE_URL}")
        write("=========================================\n")


def start_session():
    write("\n\n#################################################")
    write(f"NEW TEST SESSION - {datetime.now()}")
    write(f"API cible : {BASE_URL}")
    write("#################################################\n")


# ============================================================
# PAYLOADS
# ============================================================

def random_payload():
    def r():
        return random.choice([
            random.uniform(-1e6, 1e6),
            random.uniform(-100, 100),
            "abc",
            "",
            None
        ])

    def d():
        return random.choice([
            "2026-01-15",
            "15/01/2026",
            "invalid",
            "9999-99-99",
            ""
        ])

    return {
        "date":            d(),
        "conso_j1":        r(),
        "conso_j7":        r(),
        "conso_moy_7j":    r(),
        "temperature":     r(),
        "temperature_max": r()
    }


# ============================================================
# VERDICT ENGINE
# Seuils ajustés pour le cloud (réseau + plan gratuit Render)
# latence moyenne acceptable jusqu'à 5s sous charge
# ============================================================

def evaluate(avg_latency, errors, total):
    error_rate = errors / total
    if error_rate > 0.05 or avg_latency > 5.0:
        return "FAILED ❌"
    return "SUCCESS ✅"


# ============================================================
# FUZZ TEST
# Réduit à 100 requêtes pour respecter les limites du plan gratuit
# Pas de délai artificiel : le réseau introduit déjà de la latence
# ============================================================

def chaos_fuzz_test(n=100):
    latencies = []
    errors    = 0
    crashes   = 0

    write("===== CHAOS FUZZ TEST =====")

    for _ in range(n):
        try:
            start    = time.time()
            response = requests.post(
                f"{BASE_URL}/predict",
                json=random_payload(),
                timeout=30
            )
            latencies.append(time.time() - start)

            if response.status_code >= 500:
                crashes += 1
            elif response.status_code not in (200, 400, 422):
                errors += 1

        except Exception:
            crashes += 1

    avg     = statistics.mean(latencies) if latencies else 0
    verdict = evaluate(avg, errors + crashes, n)

    return latencies, errors, crashes, verdict


# ============================================================
# PERFORMANCE TEST
# Réduit à 50 requêtes pour le plan gratuit
# ============================================================

def performance_test(n=50):
    latencies = []

    write("\n===== PERFORMANCE TEST =====")

    payload = {
        "date":            "2026-01-15",
        "conso_j1":        62000,
        "conso_j7":        61000,
        "conso_moy_7j":    60500,
        "temperature":     4.5,
        "temperature_max": 8.0
    }

    for _ in range(n):
        start = time.time()
        requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
        latencies.append(time.time() - start)

    latencies.sort()

    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies) - 1)]

    avg     = statistics.mean(latencies)
    verdict = evaluate(avg, 0, n)

    return latencies, p95, p99, verdict


# ============================================================
# STRESS TEST
# Réduit à 50 requêtes pour le plan gratuit
# ============================================================

def stress_test(n=50):
    write("\n===== STRESS TEST =====")

    start_global = time.time()
    responses    = []
    errors       = 0

    for _ in range(n):
        payload = {
            "date":            "2026-01-15",
            "conso_j1":        random.randint(50000, 70000),
            "conso_j7":        random.randint(50000, 70000),
            "conso_moy_7j":    random.randint(50000, 70000),
            "temperature":     random.uniform(-5, 30),
            "temperature_max": random.uniform(-5, 30)
        }

        try:
            r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=30)
            responses.append(r.status_code)
            if r.status_code >= 500:
                errors += 1
        except Exception:
            errors += 1

    total_time = time.time() - start_global
    rps        = n / total_time
    verdict    = "SUCCESS ✅" if errors / n < 0.05 else "FAILED ❌"

    return responses, total_time, rps, verdict


# ============================================================
# MAIN SUITE
# ============================================================

def test_full_chaos_suite():
    write_if_new_file()
    start_session()

    lat_fuzz, err_fuzz, crash_fuzz, fuzz_verdict  = chaos_fuzz_test()
    lat_perf, p95, p99, perf_verdict              = performance_test()
    stress_res, stress_time, rps, stress_verdict  = stress_test()

    write("===== RESULT SUMMARY =====")

    write(f"Fuzz requests    : {len(lat_fuzz)}")
    write(f"Fuzz errors      : {err_fuzz}")
    write(f"Fuzz crashes     : {crash_fuzz}")
    write(f"Fuzz verdict     : {fuzz_verdict}")

    if lat_fuzz:
        write(f"Avg fuzz latency : {statistics.mean(lat_fuzz):.4f}s")

    write(f"P95 latency         : {p95:.4f}s")
    write(f"P99 latency         : {p99:.4f}s")
    write(f"Performance verdict : {perf_verdict}")

    write(f"Stress total time   : {stress_time:.2f}s")
    write(f"Requests/sec        : {rps:.2f}")
    write(f"Stress verdict      : {stress_verdict}")

    write("=========================================")
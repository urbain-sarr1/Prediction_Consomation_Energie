"""
L’ensemble des tests de charge et de stabilité réalisés (load ramp et soak test) permet 
d’évaluer de manière progressive et durable le comportement de l’API de prédiction de 
consommation face à différents niveaux de sollicitation. Le test de charge progressive 
(load ramp) consiste à augmenter graduellement le nombre de requêtes envoyées à l’API 
(10, 50, 100, 200 puis 500 requêtes) afin d’identifier à partir de quel seuil les performances 
commencent à se dégrader, notamment en observant l’évolution de la latence moyenne, du P95 et 
du taux d’erreurs. Le test de stabilité longue durée (soak test), quant à lui, simule un usage 
prolongé de l’API sur plusieurs milliers de requêtes afin de détecter d’éventuelles dégradations 
progressives des performances, des fuites mémoire ou une augmentation anormale de la latence dans 
le temps. Des points de contrôle intermédiaires permettent également de suivre l’évolution du système 
au fil de l’exécution. Dans l’ensemble, ces tests fournissent une vision claire de la capacité de 
l’API à supporter une charge croissante ainsi qu’un fonctionnement prolongé sans perte de performance 
ni instabilité.

"""


import time
import statistics
import os
from datetime import datetime
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "resultat_test")
os.makedirs(RESULT_DIR, exist_ok=True)

REPORT_FILE = os.path.join(RESULT_DIR, "rapport_charge_stabilite_tests.txt")


# ============================================================
# WRITE SYSTEM
# ============================================================

def write(line: str):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def start_session(title: str):
    write("\n#################################################")
    write(f"NEW TEST SESSION - {datetime.now()}")
    write("#################################################")
    write(f"===== {title} =====\n")


def write_if_new_file():
    if os.path.getsize(REPORT_FILE) == 0:
        write("=========================================")
        write("CHARGE / STABILITY TEST REPORT")
        write("API FASTAPI - PREDICTION CONSOMMATION")
        write("=========================================\n")


# ============================================================
# BASE PAYLOAD
# ============================================================

def base_payload():
    return {
        "date": "2026-01-15",
        "conso_j1": 62000,
        "conso_j7": 61000,
        "conso_moy_7j": 60500,
        "temperature": 4.5,
        "temperature_max": 8.0
    }


# ============================================================
# VERDICT ENGINE
# ============================================================

def evaluate_test(avg_latency, errors, total_requests):
    error_rate = errors / total_requests

    if error_rate > 0.05 or avg_latency > 0.2:
        return "FAILED ❌"
    return "SUCCESS ✅"


# ============================================================
# ⚡ LOAD RAMP TEST
# ============================================================

def run_load_ramp():
    start_session("LOAD RAMP TEST")

    steps = [10, 50, 100, 200, 500]

    ramp_results = []

    for step in steps:
        latencies = []
        errors = 0

        start_step = time.time()

        for _ in range(step):
            try:
                start = time.time()
                response = client.post("/predict", json=base_payload())
                latencies.append(time.time() - start)

                if response.status_code >= 500:
                    errors += 1

            except Exception:
                errors += 1

        total = time.time() - start_step

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]

        verdict = evaluate_test(avg, errors, step)

        write(f"[LOAD STEP {step}]")
        write(f"Avg latency: {avg:.4f}s")
        write(f"P95 latency: {p95:.4f}s")
        write(f"Errors: {errors}")
        write(f"Req/sec: {step/total:.2f}")
        write(f"VERDICT: {verdict}\n")

        ramp_results.append((step, avg, errors, verdict))

    return ramp_results


# ============================================================
# 🔁 SOAK TEST
# ============================================================

def run_soak_test(n=2000):
    start_session("SOAK TEST")

    latencies = []
    errors = 0
    checkpoint = []

    start_global = time.time()

    for i in range(n):
        try:
            start = time.time()
            response = client.post("/predict", json=base_payload())
            latencies.append(time.time() - start)
            checkpoint.append(latencies[-1])

            if response.status_code >= 500:
                errors += 1

        except Exception:
            errors += 1

        if (i + 1) % 500 == 0:
            write(f"[CHECKPOINT {i+1}] avg latency = {statistics.mean(checkpoint):.4f}s")
            checkpoint = []

    total = time.time() - start_global

    avg = statistics.mean(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    p99 = sorted(latencies)[int(0.99 * len(latencies)) - 1]

    verdict = evaluate_test(avg, errors, n)

    write("\n===== SOAK SUMMARY =====")
    write(f"Total requests: {n}")
    write(f"Avg latency: {avg:.4f}s")
    write(f"P95 latency: {p95:.4f}s")
    write(f"P99 latency: {p99:.4f}s")
    write(f"Errors: {errors}")
    write(f"Req/sec: {n/total:.2f}")
    write(f"VERDICT: {verdict}")
    write("=========================================\n")

    return avg, p95, p99, errors, verdict


# ============================================================
# MAIN SUITE
# ============================================================

def test_full_charge_suite():
    write_if_new_file()

    write("\n=========================================")
    write(f"NEW EXECUTION - {datetime.now()}")
    write("=========================================\n")

    ramp = run_load_ramp()
    soak = run_soak_test()

    avg_soak, p95, p99, errors, verdict = soak

    write("\n=========================================")
    write("FINAL RESULTTEST SUMMARY")
    write("=========================================")

    write(f"Load ramp steps: {[r[0] for r in ramp]}")
    write(f"Load ramp verdicts: {[r[3] for r in ramp]}")

    write(f"Soak avg latency: {avg_soak:.4f}s")
    write(f"Soak P95: {p95:.4f}s")
    write(f"Soak P99: {p99:.4f}s")
    write(f"Soak errors: {errors}")
    write(f"FINAL VERDICT: {verdict}")

    write("=========================================")
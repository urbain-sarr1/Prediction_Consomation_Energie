import os
import joblib
import shutil
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH  = os.path.join(ROOT_DIR, "src", "output", "dataset_journalier_enrichi.csv")
MODEL_PROD = os.path.join(ROOT_DIR, "src", "output", "model_prod.joblib")
MODEL_V2   = os.path.join(ROOT_DIR, "src", "output", "model_v2.joblib")
CSV_PROD   = os.path.join(ROOT_DIR, "src", "output", "resultats_optimisation.csv")
CSV_V2     = os.path.join(ROOT_DIR, "src", "output", "resultats_optimisation_v2.csv")

REPORT_DIR  = os.path.join(ROOT_DIR, "test", "resultat_test")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_FILE = os.path.join(REPORT_DIR, "rapport_performance_test.txt")

ANNEE_TEST  = 2023


# ============================================================
# WRITE
# ============================================================

def write(line: str):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# ÉVALUATION DIRECTE D'UN MODÈLE
# Chaque modèle est évalué avec SES PROPRES features
# pour éviter le mismatch quand prod et v2 n'ont pas le même nb de features
# ============================================================

def evaluer_modele(modele, features, data, y_test):
    predictions = modele.predict(data[features])
    r2   = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    return r2, mape, predictions


# ============================================================
# TEST
# ============================================================

def test_performance_api(mode: str = "drift"):
    """
    mode='retrain' : compare model_prod vs model_v2 directement (joblib)
                     chaque modèle est évalué avec ses propres features
                     si v2 gagne → CSV provisoire promu en CSV stable

    mode='drift'   : compare model_prod en live vs ses métriques CSV enregistrées
                     détecte une dérive des données (pas du modèle)
    """

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    write("\n" + "=" * 60)
    write(f"RUN DATE : {run_time}  |  MODE : {mode.upper()}")
    write("=" * 60 + "\n")

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    assert "Date" in df.columns, "Colonne 'Date' manquante dans le dataset."

    TARGET = "Consommation"
    test   = df[df["Annee"] >= ANNEE_TEST].dropna(subset=[TARGET])

    assert len(test) > 0, f"Dataset vide après filtre Annee >= {ANNEE_TEST}"

    write(f"Période test  : Annee >= {ANNEE_TEST}")
    write(f"Jours de test : {len(test)}\n")

    # ----------------------------------------------------------
    # CHARGEMENT model_prod — évalué avec SES propres features
    # ----------------------------------------------------------
    assert os.path.exists(MODEL_PROD), f"model_prod introuvable : {MODEL_PROD}"
    bundle_prod   = joblib.load(MODEL_PROD)
    modele_prod   = bundle_prod["model"]
    features_prod = bundle_prod["features"]   # ← features propres à prod

    y_test = test[TARGET]
    r2_prod, mape_prod, _ = evaluer_modele(modele_prod, features_prod, test, y_test)

    write("--- MODEL PROD (en cours) ---")
    write(f"Features : {len(features_prod)}")
    write(f"R2       : {r2_prod:.4f}")
    write(f"MAPE     : {mape_prod:.4f} %\n")

    # ==========================================================
    # MODE RETRAIN : prod vs v2
    # ==========================================================
    if mode == "retrain":

        assert os.path.exists(MODEL_V2), f"model_v2 introuvable : {MODEL_V2}"
        bundle_v2   = joblib.load(MODEL_V2)
        modele_v2   = bundle_v2["model"]
        features_v2 = bundle_v2["features"]   # ← features propres à v2

        r2_v2, mape_v2, _ = evaluer_modele(modele_v2, features_v2, test, y_test)

        r2_drop   = ((r2_prod - r2_v2) / r2_prod) * 100 if r2_prod != 0 else float("inf")
        mape_diff = mape_v2 - mape_prod

        write("--- MODEL V2 (candidat) ---")
        write(f"Features : {len(features_v2)}")
        write(f"R2       : {r2_v2:.4f}")
        write(f"MAPE     : {mape_v2:.4f} %\n")

        write("--- ÉCARTS V2 vs PROD ---")
        write(f"R2 amélioration (%)     : {-r2_drop:.2f}")
        write(f"MAPE variation (points) : {mape_diff:.2f}  (seuil critique : > 2 pts)\n")

        if mape_diff < 0 and r2_drop < 5:
            verdict = "✅ V2 MEILLEUR — déploiement autorisé"
            result  = 0

            if os.path.exists(CSV_V2):
                shutil.copy2(CSV_V2, CSV_PROD)
                os.remove(CSV_V2)
                write("📄 resultats_optimisation.csv mis à jour avec les métriques v2\n")

        else:
            verdict = "❌ V2 NON MEILLEUR — déploiement annulé, prod conservé"
            result  = 1

            if os.path.exists(CSV_V2):
                os.remove(CSV_V2)
                write("🗑️  resultats_optimisation_v2.csv supprimé (v2 non retenu)\n")

    # ==========================================================
    # MODE DRIFT : prod en live vs ses métriques CSV enregistrées
    # ==========================================================
    elif mode == "drift":

        assert os.path.exists(CSV_PROD), f"CSV prod introuvable : {CSV_PROD}"
        csv = pd.read_csv(CSV_PROD)
        xgb = csv[csv["Modèle"] == "XGBoost"]
        assert len(xgb) > 0, "XGBoost introuvable dans resultats_optimisation.csv"
        xgb = xgb.iloc[0]

        ref_r2   = float(xgb["R2_test"])
        ref_mape = float(xgb["MAPE_test_%"])

        r2_drop   = ((ref_r2 - r2_prod) / ref_r2) * 100 if ref_r2 != 0 else float("inf")
        mape_diff = mape_prod - ref_mape

        write("--- RÉFÉRENCE CSV (métriques enregistrées de prod) ---")
        write(f"R2 ref   : {ref_r2:.4f}")
        write(f"MAPE ref : {ref_mape:.4f} %\n")

        write("--- ÉCARTS LIVE vs RÉFÉRENCE ---")
        write(f"R2 drop (%)             : {r2_drop:.2f}  (seuil : > 5 %)")
        write(f"MAPE variation (points) : {mape_diff:.2f}  (seuil : > 2 pts)\n")

        if r2_drop > 5 or abs(mape_diff) > 2:
            verdict = "⚠️  DRIFT DÉTECTÉ — données potentiellement dégradées, relancer un retrain"
            result  = 1
        else:
            verdict = "✅ PAS DE DRIFT — modèle prod stable sur les données actuelles"
            result  = 0

    else:
        raise ValueError(f"Mode inconnu : '{mode}'. Utiliser 'retrain' ou 'drift'.")

    write("--- VERDICT ---")
    write(verdict)
    write("=" * 60 + "\n")

    return result
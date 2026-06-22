import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ============================================================
# PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH        = os.path.join(ROOT_DIR, "src", "output", "dataset_journalier_enrichi.csv")
MODEL_PROD       = os.path.join(ROOT_DIR, "src", "output", "model_prod.joblib")
MODEL_V2         = os.path.join(ROOT_DIR, "src", "output", "model_v2.joblib")

# CSV stable = métriques du modèle actuellement en prod
CSV_PROD         = os.path.join(ROOT_DIR, "src", "output", "resultats_optimisation.csv")
# CSV provisoire = métriques du nouveau modèle v2 candidat
CSV_V2           = os.path.join(ROOT_DIR, "src", "output", "resultats_optimisation_v2.csv")

REPORT_DIR       = os.path.join(ROOT_DIR, "test", "resultat_test")
os.makedirs(REPORT_DIR, exist_ok=True)
REPORT_FILE      = os.path.join(REPORT_DIR, "rapport_performance_test.txt")

ANNEE_TEST       = 2023


# ============================================================
# WRITE
# ============================================================

def write(line: str):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# ÉVALUATION DIRECTE D'UN MODÈLE
# ============================================================

def evaluer_modele(modele, features, data, y_test):
    predictions = modele.predict(data[features])
    r2   = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    return r2, mape, predictions


# ============================================================
# TEST — deux modes :
#   mode "retrain" : compare model_prod vs model_v2 (joblib direct)
#   mode "drift"   : compare model_prod vs ses métriques CSV enregistrées
# ============================================================

def test_performance_api(mode: str = "drift"):
    """
    mode='retrain' : appelé après optimisation_hyperparametres()
                     compare model_prod vs model_v2 directement
                     si v2 gagne → CSV provisoire devient CSV stable

    mode='drift'   : appelé seul, sans retrain
                     compare model_prod en live vs ses métriques CSV enregistrées
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
    # CHARGEMENT model_prod (toujours nécessaire)
    # ----------------------------------------------------------
    assert os.path.exists(MODEL_PROD), f"model_prod introuvable : {MODEL_PROD}"
    bundle_prod = joblib.load(MODEL_PROD)
    modele_prod = bundle_prod["model"]
    features    = bundle_prod["features"]

    y_test = test[TARGET]
    r2_prod, mape_prod, _ = evaluer_modele(modele_prod, features, test, y_test)

    write("--- MODEL PROD (en cours) ---")
    write(f"R2   : {r2_prod:.4f}")
    write(f"MAPE : {mape_prod:.4f} %\n")

    # ==========================================================
    # MODE RETRAIN : prod vs v2 en direct
    # ==========================================================
    if mode == "retrain":

        assert os.path.exists(MODEL_V2), f"model_v2 introuvable : {MODEL_V2}"
        bundle_v2   = joblib.load(MODEL_V2)
        modele_v2   = bundle_v2["model"]

        r2_v2, mape_v2, _ = evaluer_modele(modele_v2, features, test, y_test)

        r2_drop   = ((r2_prod - r2_v2) / r2_prod) * 100 if r2_prod != 0 else float("inf")
        mape_diff = mape_v2 - mape_prod

        write("--- MODEL V2 (candidat) ---")
        write(f"R2   : {r2_v2:.4f}")
        write(f"MAPE : {mape_v2:.4f} %\n")

        write("--- ÉCARTS V2 vs PROD ---")
        write(f"R2 amélioration (%)     : {-r2_drop:.2f}")
        write(f"MAPE variation (points) : {mape_diff:.2f}  (seuil critique : > 2 pts)\n")

        # V2 gagne si sa MAPE est inférieure à celle de prod
        # et que le R2 ne chute pas de plus de 5 %
        if mape_diff < 0 and r2_drop < 5:
            verdict = "✅ V2 MEILLEUR — déploiement autorisé"
            result  = 0

            # CSV provisoire → CSV stable (écrase l'ancien)
            if os.path.exists(CSV_V2):
                import shutil
                shutil.copy2(CSV_V2, CSV_PROD)
                os.remove(CSV_V2)
                write("📄 resultats_optimisation.csv mis à jour avec les métriques v2\n")

        else:
            verdict = "❌ V2 NON MEILLEUR — déploiement annulé, prod conservé"
            result  = 1

            # CSV provisoire supprimé, CSV stable intact
            if os.path.exists(CSV_V2):
                os.remove(CSV_V2)
                write("🗑️  resultats_optimisation_v2.csv supprimé (v2 non retenu)\n")

    # ==========================================================
    # MODE DRIFT : prod en live vs ses métriques enregistrées
    # ==========================================================
    elif mode == "drift":

        assert os.path.exists(CSV_PROD), f"CSV prod introuvable : {CSV_PROD}"
        csv    = pd.read_csv(CSV_PROD)
        xgb    = csv[csv["Modèle"] == "XGBoost"]
        assert len(xgb) > 0, "XGBoost introuvable dans resultats_optimisation.csv"
        xgb    = xgb.iloc[0]

        ref_r2   = float(xgb["R2_test"])
        ref_mape = float(xgb["MAPE_test_%"])

        r2_drop   = ((ref_r2 - r2_prod) / ref_r2) * 100 if ref_r2 != 0 else float("inf")
        mape_diff = r2_prod - ref_mape   # écart live vs référence enregistrée

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
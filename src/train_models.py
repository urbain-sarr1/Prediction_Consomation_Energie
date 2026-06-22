import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

try:
    from xgboost import XGBRegressor
    XGBOOST_DISPO = True
except ImportError:
    XGBOOST_DISPO = False
    print("(XGBoost non installé -> ignoré. Pour le tester : pip install xgboost)\n")

# ============================================================
# ENTRAÎNEMENT — 4 modèles, en V1 (avec Prevision_J1) et V2 (sans)
# Métriques : R², RMSE, MAPE, temps d'apprentissage
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT = os.path.join(OUT_DIR, "dataset_journalier_enrichi.csv")
print(f"Lecture du dataset enrichi : {INPUT}\n")

TARGET = "Consommation"

FEATURES_V2 = [
    "Conso_J1", "Conso_J7", "Conso_moy_7j",
    "Temperature", "Temperature_max", "Mois_sin", "Mois_cos",
    "Jour_semaine_sin", "Jour_semaine_cos",
    "Est_weekend", "Est_ferie",
]
FEATURES_V1 = ["Prevision_J1"] + FEATURES_V2

ANNEE_TRAIN = 2022   # train : années < 2022
ANNEE_VAL   = 2022   # val   : année == 2022
ANNEE_TEST  = 2023   # test  : années >= 2023

# ------------------------------------------------------------
df = pd.read_csv(INPUT).dropna(subset=set(FEATURES_V1 + [TARGET]))


def construire_modeles():
    modeles = {
        "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
        "Arbre de décision": DecisionTreeRegressor(random_state=42),
        "KNN":               make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        "MLP":               make_pipeline(StandardScaler(),
                                           MLPRegressor(hidden_layer_sizes=(64, 32),
                                                        max_iter=1000, random_state=42)),
    }
    if XGBOOST_DISPO:
        modeles["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6,
                                          learning_rate=0.1, random_state=42, verbosity=0)
    return modeles


def evaluer(features, label):
    train = df[df["Annee"] <  ANNEE_TRAIN]
    val   = df[df["Annee"] == ANNEE_VAL]
    test  = df[df["Annee"] >= ANNEE_TEST]

    if len(val) == 0:
        raise ValueError(f"Aucune donnée de validation (année {ANNEE_VAL}).")
    if len(test) == 0:
        raise ValueError(f"Aucune donnée de test (>= {ANNEE_TEST}).")

    X_train, y_train = train[features], train[TARGET]
    X_val,   y_val   = val[features],   val[TARGET]
    X_test,  y_test  = test[features],  test[TARGET]

    lignes = []
    modeles = construire_modeles()
    for nom, modele in modeles.items():
        t0 = time.time()
        modele.fit(X_train, y_train)
        duree = time.time() - t0

        pred_val  = modele.predict(X_val)
        pred_test = modele.predict(X_test)

        lignes.append({
            "Version":    label,
            "Modèle":     nom,
            # Validation
            "Val_R2":     r2_score(y_val, pred_val),
            "Val_RMSE":   np.sqrt(mean_squared_error(y_val, pred_val)),
            "Val_MAPE_%": mean_absolute_percentage_error(y_val, pred_val) * 100,
            # Test
            "R2":         r2_score(y_test, pred_test),
            "RMSE":       np.sqrt(mean_squared_error(y_test, pred_test)),
            "MAPE_%":     mean_absolute_percentage_error(y_test, pred_test) * 100,
            "Temps_s":    duree,
        })
    return pd.DataFrame(lignes)


print(f"Train : {(df['Annee'] < ANNEE_TRAIN).sum():,} jours  |  "
      f"Val   : {(df['Annee'] == ANNEE_VAL).sum():,} jours  |  "
      f"Test  : {(df['Annee'] >= ANNEE_TEST).sum():,} jours\n")

resultats = pd.concat([
    evaluer(FEATURES_V1, "V1 (avec Prevision_J1)"),
    evaluer(FEATURES_V2, "V2 (sans Prevision_J1)"),
], ignore_index=True)

print("=== Comparaison des modèles ===")
print(resultats.round(3).to_string(index=False))

csv_out = os.path.join(OUT_DIR, "resultats_modeles.csv")
resultats.to_csv(csv_out, index=False, encoding="utf-8-sig")
print(f"\nRésultats -> {csv_out}")

# ------------------------------------------------------------
# Sauvegarde du meilleur modèle V2 (sélection sur Val_MAPE_%,
# réentraîné sur train + val avant export)
res_v2 = resultats[resultats["Version"].str.startswith("V2")]
meilleur = res_v2.sort_values("Val_MAPE_%").iloc[0]["Modèle"]

train_val = df[df["Annee"] < ANNEE_TEST]   # train + val réunis pour l'export final
modele_final = construire_modeles()[meilleur]
modele_final.fit(train_val[FEATURES_V2], train_val[TARGET])

model_path = os.path.join(OUT_DIR, "model_v1.joblib")
joblib.dump({"model": modele_final, "features": FEATURES_V2, "target": TARGET}, model_path)
print(f"Meilleur modèle sans Prevision_J1 ({meilleur}) sauvegardé -> {model_path}")
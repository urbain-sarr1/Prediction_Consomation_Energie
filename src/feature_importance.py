import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# FEATURE IMPORTANCE — quelles variables comptent vraiment ?
# On entraîne un Random Forest et on lit l'importance de chaque variable.
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

INPUT = os.path.join(OUT_DIR, "dataset_journalier_enrichi.csv")
print(f"Lecture du dataset enrichi : {INPUT}")

TARGET = "Consommation"

# Variables candidates testées par la feature importance.
# On ne met QUE des variables connues au moment de prédire.
# La température du jour est admise comme proxy de la prévision météo à J-1
# (disponible la veille). On exclut en revanche les valeurs du jour même
# non prévisibles (Nucleaire, Eolien, Solaire... -> fuite de données).
CANDIDATES = [
    "Prevision_J1",
    "Conso_J1", "Conso_J7",
    "Conso_moy_3j", "Conso_moy_7j",
    "Temperature","Temperature_min", "Temperature_max",
    "Mois_sin", "Mois_cos", "Jour_semaine_sin", "Jour_semaine_cos",
    "Est_weekend", "Est_ferie",
    "Tempo",
    "Is_Covid", "Annee",
]

# ------------------------------------------------------------
df = pd.read_csv(INPUT)


def feature_importance(df, features, titre, suffixe):
    """Entraîne un Random Forest et sort le classement d'importance des variables."""
    features = [c for c in features if c in df.columns]
    data = df.dropna(subset=features + [TARGET])

    # Entraînement sur le passé uniquement (split chronologique propre)
    train = data[data["Annee"] <= 2021] if "Annee" in data.columns else data

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(train[features], train[TARGET])

    importances = (
        pd.Series(rf.feature_importances_, index=features)
        .sort_values(ascending=False)
    )

    print(f"\n=== {titre} ===")
    print(importances.round(4).to_string())

    csv_out = os.path.join(OUT_DIR, f"feature_importance_{suffixe}.csv")
    importances.to_csv(csv_out, header=["importance"])

    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind="barh", color="steelblue")
    plt.title(titre)
    plt.xlabel("Importance")
    plt.tight_layout()
    png_out = os.path.join(OUT_DIR, f"feature_importance_{suffixe}.png")
    plt.savefig(png_out, dpi=120)
    plt.close()

    print(f"  -> {csv_out}")
    print(f"  -> {png_out}")
    return importances


# 1) AVEC Prevision_J1 : montre qu'elle domine (justifie le V1/V2)
feature_importance(
    df, CANDIDATES,
    "Feature importance — AVEC Prevision_J1",
    "avec_prevision",
)

# 2) SANS Prevision_J1 : révèle le vrai classement de TES variables (features V2)
feature_importance(
    df, [c for c in CANDIDATES if c != "Prevision_J1"],
    "Feature importance — SANS Prevision_J1",
    "sans_prevision",
)
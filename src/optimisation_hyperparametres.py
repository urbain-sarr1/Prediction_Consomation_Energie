def optimisation_hyperparametres():

    import pandas as pd
    import numpy as np
    import os
    import joblib

    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

    try:
        from xgboost import XGBRegressor
        XGBOOST_DISPO = True
    except ImportError:
        XGBOOST_DISPO = False
        print("(XGBoost non installé -> ignoré. Pour le tester : pip install xgboost)\n")

    # ============================================================
    # OPTIMISATION DES HYPERPARAMÈTRES (sur les features V2)
    # Validation croisée temporelle (TimeSeriesSplit) -> pas de fuite passé/futur
    # ============================================================
    HERE = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(HERE, "output")
    os.makedirs(OUT_DIR, exist_ok=True)

    INPUT = os.path.join(OUT_DIR, "dataset_journalier_enrichi.csv")
    print(f"Lecture du dataset enrichi : {INPUT}\n")

    TARGET = "Consommation"
    FEATURES_V2 = [
        "Conso_J1", "Conso_J7", "Conso_moy_7j",
        "Temperature", "Temperature_max",
        "Jour_semaine_sin", "Jour_semaine_cos",
        "Est_weekend", "Est_ferie",
    ]

    ANNEE_TRAIN = 2022   # train : années < 2022
    ANNEE_VAL   = 2022   # val   : année == 2022
    ANNEE_TEST  = 2023   # test  : années >= 2023

    df = pd.read_csv(INPUT).dropna(subset=FEATURES_V2 + [TARGET])
    if "Date" in df.columns:
        df = df.sort_values("Date")

    train   = df[df["Annee"] <  ANNEE_TRAIN]
    val     = df[df["Annee"] == ANNEE_VAL]
    test    = df[df["Annee"] >= ANNEE_TEST]

    if len(val) == 0:
        raise ValueError(f"Aucune donnée de validation (année {ANNEE_VAL}).")
    if len(test) == 0:
        raise ValueError(f"Aucune donnée de test (>= {ANNEE_TEST}).")

    X_train, y_train = train[FEATURES_V2], train[TARGET]
    X_val,   y_val   = val[FEATURES_V2],   val[TARGET]
    X_test,  y_test  = test[FEATURES_V2],  test[TARGET]

    print(f"Train : {len(train):,} jours  |  Val : {len(val):,} jours  |  Test : {len(test):,} jours\n")

    GRILLES = {
        "Random Forest": (
            Pipeline([("model", RandomForestRegressor(random_state=42))]),
            {
                "model__n_estimators": [100, 200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 2, 5],
            },
        ),
        "Gradient Boosting": (
            Pipeline([("model", HistGradientBoostingRegressor(random_state=42))]),
            {
                "model__max_iter": [200, 400],
                "model__max_depth": [None, 5, 10],
                "model__learning_rate": [0.05, 0.1],
            },
        ),
        "Arbre de décision": (
            Pipeline([("model", DecisionTreeRegressor(random_state=42))]),
            {
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_leaf": [1, 5, 10],
            },
        ),
        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
            {
                "model__n_neighbors": [3, 5, 7, 11, 15],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "MLP": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", MLPRegressor(max_iter=1000, random_state=42))]),
            {
                "model__hidden_layer_sizes": [(64, 32), (128, 64), (100,)],
                "model__alpha": [0.0001, 0.001],
            },
        ),
    }

    if XGBOOST_DISPO:
        GRILLES["XGBoost"] = (
            Pipeline([("model", XGBRegressor(random_state=42, verbosity=0))]),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 6, 10],
                "model__learning_rate": [0.05, 0.1],
            },
        )

    cv = TimeSeriesSplit(n_splits=5)
    scoring = "neg_mean_absolute_percentage_error"

    resultats = []
    meilleur_global = {"mape": np.inf}
    modeles_optimises = {}

    for nom, (pipe, grille) in GRILLES.items():
        print(f"Optimisation : {nom} ...")

        # GridSearchCV s'entraîne sur train, sélectionne sur val via TimeSeriesSplit
        search = GridSearchCV(pipe, grille, scoring=scoring, cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)

        best = search.best_estimator_

        # Sélection du meilleur modèle sur val (pas de fuite du test)
        pred_val  = best.predict(X_val)
        mape_val  = mean_absolute_percentage_error(y_val, pred_val) * 100

        # Évaluation finale sur test (lecture seule, ne sert pas à la sélection)
        pred_test = best.predict(X_test)
        mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100

        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        resultats.append({
            "Modèle":        nom,
            "R2_val":        r2_score(y_val, pred_val),
            "RMSE_val":      np.sqrt(mean_squared_error(y_val, pred_val)),
            "MAPE_val_%":    mape_val,
            "R2_test":       r2_score(y_test, pred_test),
            "RMSE_test":     np.sqrt(mean_squared_error(y_test, pred_test)),
            "MAPE_test_%":   mape_test,
            "Meilleurs_params": params,
        })

        # Sélection sur val uniquement
        if mape_val < meilleur_global["mape"]:
            meilleur_global = {"mape": mape_val, "nom": nom, "modele": best}

        modeles_optimises[nom] = {"modele": best, "mape_val": mape_val}

    resultats = pd.DataFrame(resultats).sort_values("MAPE_val_%").reset_index(drop=True)

    csv_out = os.path.join(OUT_DIR, "resultats_optimisation.csv")
    resultats.to_csv(csv_out, index=False, encoding="utf-8-sig")

    MODELE_PRODUCTION = "XGBoost"

    if MODELE_PRODUCTION in modeles_optimises:
        nom_final = MODELE_PRODUCTION
    else:
        nom_final = meilleur_global["nom"]

    modele_final = modeles_optimises[nom_final]["modele"]

    # Réentraînement final sur train + val avant export
    train_val = df[df["Annee"] < ANNEE_TEST]
    modele_final.fit(train_val[FEATURES_V2], train_val[TARGET])

    model_path = os.path.join(OUT_DIR, "model_v2.joblib")
    joblib.dump({"model": modele_final, "features": FEATURES_V2, "target": TARGET}, model_path)


# ============================================================
# MAIN (IMPORTANT)
# ============================================================

if __name__ == "__main__":
    optimisation_hyperparametres()
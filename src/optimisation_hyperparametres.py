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
    # CONFIG
    # ============================================================
    HERE    = os.path.dirname(os.path.abspath(__file__))
    OUT_DIR = os.path.join(HERE, "output")
    os.makedirs(OUT_DIR, exist_ok=True)

    INPUT  = os.path.join(OUT_DIR, "dataset_journalier_enrichi.csv")
    print(f"Lecture du dataset enrichi : {INPUT}\n")

    TARGET      = "Consommation"
    FEATURES_V2 = [
        "Conso_J1", "Conso_J7", "Conso_moy_7j",
        "Temperature", "Temperature_max",
        "Mois_sin", "Mois_cos",
        "Jour_semaine_sin", "Jour_semaine_cos",
        "Est_weekend", "Est_ferie",
    ]

    ANNEE_TRAIN = 2022
    ANNEE_VAL   = 2022
    ANNEE_TEST  = 2023

    df = pd.read_csv(INPUT).dropna(subset=FEATURES_V2 + [TARGET])
    if "Date" in df.columns:
        df = df.sort_values("Date")

    train = df[df["Annee"] <  ANNEE_TRAIN]
    val   = df[df["Annee"] == ANNEE_VAL]
    test  = df[df["Annee"] >= ANNEE_TEST]

    if len(val) == 0:
        raise ValueError(f"Aucune donnée de validation (année {ANNEE_VAL}).")
    if len(test) == 0:
        raise ValueError(f"Aucune donnée de test (>= {ANNEE_TEST}).")

    X_train, y_train = train[FEATURES_V2], train[TARGET]
    X_val,   y_val   = val[FEATURES_V2],   val[TARGET]
    X_test,  y_test  = test[FEATURES_V2],  test[TARGET]

    print(f"Train : {len(train):,} jours  |  Val : {len(val):,} jours  |  Test : {len(test):,} jours\n")

    # ============================================================
    # GRILLES ÉLARGIES
    # ============================================================
    GRILLES = {

        "Random Forest": (
            Pipeline([("model", RandomForestRegressor(random_state=42, n_jobs=-1))]),
            {
                "model__n_estimators":    [100, 200, 400, 600],
                "model__max_depth":       [None, 10, 20, 30],
                "model__min_samples_leaf":[1, 2, 5, 10],
                "model__max_features":    ["sqrt", "log2", 0.8],
                "model__min_samples_split": [2, 5, 10],
            },
        ),

        "Gradient Boosting": (
            Pipeline([("model", HistGradientBoostingRegressor(random_state=42))]),
            {
                "model__max_iter":        [200, 400, 600],
                "model__max_depth":       [None, 5, 10, 15],
                "model__learning_rate":   [0.01, 0.05, 0.1, 0.2],
                "model__min_samples_leaf":[10, 20, 30],
                "model__l2_regularization":[0.0, 0.1, 1.0],
            },
        ),

        "Arbre de décision": (
            Pipeline([("model", DecisionTreeRegressor(random_state=42))]),
            {
                "model__max_depth":        [None, 5, 10, 20, 30],
                "model__min_samples_leaf": [1, 5, 10, 20],
                "model__min_samples_split":[2, 5, 10, 20],
                "model__max_features":     [None, "sqrt", "log2"],
            },
        ),

        "KNN": (
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor())]),
            {
                "model__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
                "model__weights":     ["uniform", "distance"],
                "model__metric":      ["euclidean", "manhattan", "minkowski"],
                "model__p":           [1, 2],
            },
        ),

        "MLP": (
            Pipeline([("scaler", StandardScaler()),
                      ("model", MLPRegressor(max_iter=2000, random_state=42, early_stopping=True))]),
            {
                "model__hidden_layer_sizes": [(64, 32), (128, 64), (256, 128), (128, 64, 32), (100,), (200,)],
                "model__alpha":              [0.0001, 0.001, 0.01],
                "model__learning_rate_init": [0.001, 0.01],
                "model__activation":         ["relu", "tanh"],
            },
        ),
    }

    if XGBOOST_DISPO:
        GRILLES["XGBoost"] = (
            Pipeline([("model", XGBRegressor(random_state=42, verbosity=0, n_jobs=-1))]),
            {
                "model__n_estimators":  [200, 400, 600, 800],
                "model__max_depth":     [3, 6, 8, 10],
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__subsample":     [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 1.0],
                "model__reg_alpha":     [0, 0.1, 1.0],
                "model__reg_lambda":    [1.0, 2.0, 5.0],
            },
        )

    # ============================================================
    # OPTIMISATION
    # ============================================================
    cv      = TimeSeriesSplit(n_splits=5)
    scoring = "neg_mean_absolute_percentage_error"

    resultats        = []
    meilleur_global  = {"mape": np.inf}
    modeles_optimises = {}

    for nom, (pipe, grille) in GRILLES.items():
        nb_combos = 1
        for v in grille.values():
            nb_combos *= len(v)
        print(f"Optimisation : {nom} ... ({nb_combos} combinaisons × 5 folds)")

        search = GridSearchCV(pipe, grille, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)

        best = search.best_estimator_

        pred_val  = best.predict(X_val)
        mape_val  = mean_absolute_percentage_error(y_val, pred_val) * 100

        pred_test = best.predict(X_test)
        mape_test = mean_absolute_percentage_error(y_test, pred_test) * 100

        params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        resultats.append({
            "Modèle":           nom,
            "R2_val":           r2_score(y_val, pred_val),
            "RMSE_val":         np.sqrt(mean_squared_error(y_val, pred_val)),
            "MAPE_val_%":       mape_val,
            "R2_test":          r2_score(y_test, pred_test),
            "RMSE_test":        np.sqrt(mean_squared_error(y_test, pred_test)),
            "MAPE_test_%":      mape_test,
            "Meilleurs_params": params,
        })

        print(f"  → MAPE val : {mape_val:.4f}%  |  MAPE test : {mape_test:.4f}%  |  params : {params}")

        if mape_val < meilleur_global["mape"]:
            meilleur_global = {"mape": mape_val, "nom": nom, "modele": best}

        modeles_optimises[nom] = {"modele": best, "mape_val": mape_val}

    resultats = pd.DataFrame(resultats).sort_values("MAPE_val_%").reset_index(drop=True)

    csv_out = os.path.join(OUT_DIR, "resultats_optimisation_v2.csv")
    resultats.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\nRésultats provisoires -> {csv_out}")

    # ============================================================
    # SÉLECTION DU MODÈLE FINAL
    # Par défaut XGBoost si dispo, sinon le meilleur sur val
    # ============================================================
    MODELE_PRODUCTION = "XGBoost"

    if MODELE_PRODUCTION in modeles_optimises:
        nom_final = MODELE_PRODUCTION
    else:
        nom_final = meilleur_global["nom"]

    print(f"\nModèle sélectionné : {nom_final}")

    modele_final = modeles_optimises[nom_final]["modele"]

    # Réentraînement final sur train + val avant export
    train_val = df[df["Annee"] < ANNEE_TEST]
    modele_final.fit(train_val[FEATURES_V2], train_val[TARGET])

    model_path = os.path.join(OUT_DIR, "model_v2.joblib")
    joblib.dump({"model": modele_final, "features": FEATURES_V2, "target": TARGET}, model_path)
    print(f"model_v2 sauvegardé -> {model_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    optimisation_hyperparametres()
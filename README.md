# Prédiction de la Consommation Électrique 🇫🇷⚡

API de prédiction de la consommation électrique journalière en France, basée sur des données RTE Éco2mix (2014–2025) et déployée sur le cloud via Render.

---

## 🏗️ Architecture du projet
DATA (RTE Éco2mix)

↓

Préparation & Feature Engineering

↓

Optimisation des hyperparamètres (GridSearchCV + TimeSeriesSplit)

↓

Évaluation (Train / Val / Test)

↓

Swap sécurisé des modèles (prod → backup → v2 → prod)

↓

API FastAPI conteneurisée (Docker)

↓

Déploiement cloud (Render)

↓

Monitoring post-déploiement

---

## 📁 Structure du projet
CODE/

├── src/

│   ├── api.py                          # API FastAPI

│   ├── prepare_dataset.py              # Préparation du dataset

│   ├── train_models.py                 # Entraînement des modèles

│   ├── optimisation_hyperparametres.py # Optimisation GridSearchCV

│   ├── retrain.py                      # Pipeline de réentraînement

│   ├── deploy_cloud.py                 # Déploiement sur Render

│   └── output/

│       ├── dataset_journalier_enrichi.csv

│       ├── model_prod.joblib

│       ├── model_v2.joblib

│       ├── model_backup.joblib

│       └── resultats_optimisation.csv

├── test/

│   ├── test_navigation.py              # Tests des endpoints API

│   ├── test_fuzz_chaos.py              # Tests chaos / fuzz / stress

│   ├── test_charge_stabilite.py        # Tests charge et stabilité

│   ├── test_business_logic.py          # Tests cohérence des prédictions

│   ├── test_performances.py            # Comparaison prod vs v2

│   ├── test_monitoring_prod.py         # Monitoring post-déploiement

│   └── resultat_test/                  # Rapports générés automatiquement

├── Dockerfile

├── docker-compose.yml

├── render.yaml

└── requirements.txt

---

## ⚙️ Étape 1 — Préparation du dataset

Les données sont téléchargeables depuis [RTE Éco2mix](https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/telecharger-indicateurs).

```bash
python src/prepare_dataset.py
```

**Ce que fait ce script :**
- Télécharge et fusionne les fichiers RTE (2014–2025)
- Conserve uniquement les colonnes `Date`, `Heure` et `Consommation` pour éviter tout data leakage
- Nettoie les données et crée une colonne datetime
- Regroupe par jour et calcule la consommation moyenne journalière
- Crée des variables calendaires (`Mois_sin`, `Mois_cos`, `Jour_semaine_sin`, `Jour_semaine_cos`, `Est_weekend`, `Est_ferie`)
- Crée des variables historiques (`Conso_J1`, `Conso_J7`, `Conso_moy_7j`)
- Sauvegarde le dataset final dans `src/output/dataset_journalier_enrichi.csv`

> Les variables de production énergétique (nucléaire, gaz, éolien) ont été exclues car elles correspondent à des valeurs observées non disponibles au moment de la prédiction.

---

## 🤖 Étape 2 — Entraînement des modèles

```bash
python src/train_models.py
```

**Modèles comparés :**

| Modèle | R² | RMSE | MAPE |
|---|---|---|---|
| Random Forest | 0.972 | 1511 | 1.99% |
| Gradient Boosting | - | - | - |
| Arbre de décision | - | - | - |
| KNN | - | - | - |
| MLP | - | - | - |
| **XGBoost** ✅ | **0.978** | - | **~2%** |

**Découpage temporel :**
- Train : `Annee < 2022`
- Validation : `Annee == 2022`
- Test : `Annee >= 2023`

---

## 🔧 Étape 3 — Optimisation des hyperparamètres

```bash
python src/optimisation_hyperparametres.py
```

- Utilise `GridSearchCV` avec `TimeSeriesSplit` (5 folds) pour éviter toute fuite d'information
- Sélection du meilleur modèle sur la validation (`MAPE_val`)
- Évaluation finale sur le jeu de test (lecture seule)
- Sauvegarde du modèle optimisé dans `src/output/model_v2.joblib`

---

## 🔄 Étape 4 — Pipeline de réentraînement

```bash
python src/retrain.py
```

**Le pipeline effectue automatiquement :**
STEP 1 → Optimisation des hyperparamètres  → model_v2.joblib

STEP 2 → Comparaison prod vs v2 (R², MAPE)

STEP 3 → Si v2 meilleur → swap sécurisé des modèles

prod   → backup

v2     → prod

backup → v2

Si v2 non meilleur → prod conservé

> Le swap utilise des fichiers `.tmp` intermédiaires pour garantir qu'aucun modèle n'est perdu même en cas de crash.

---

## 🚀 Étape 5 — API FastAPI

**Lancer en local :**
```bash
uvicorn src.api:app --reload
```

**Endpoints disponibles :**

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interface de démonstration |
| `GET` | `/health` | État du service et du modèle |
| `POST` | `/predict` | Prédiction de la consommation |

**Exemple de requête `/predict` :**
```json
{
  "date": "2025-10-01",
  "conso_j1": 46217.92,
  "conso_j7": 46122.92,
  "conso_moy_7j": 46263.90,
  "temperature": 13.74,
  "temperature_max": 19.25
}
```

**Réponse :**
```json
{
  "date": "2025-10-01",
  "consommation_prevue_MW": 46898.0
}
```

> Vraie consommation ce jour : **46 769 MW** → écart de **0.27%** ✅

---

## 🐳 Étape 6 — Docker

**Build et lancement :**
```bash
docker-compose up -d
```

**Arrêt :**
```bash
docker-compose down
```

L'API sera accessible sur `http://localhost:8000`.

---

## ☁️ Étape 7 — Déploiement cloud (Render)

**Déployer après un réentraînement :**
```bash
python src/deploy_cloud.py "new model prod"
```

Ce script effectue automatiquement :
git add src/output/model_prod.joblib

git commit -m "..."

git push origin main   → Render rebuild et redémarre automatiquement

**API en production :**
https://api-prediction-consomation-energie.onrender.com

---

## 🧪 Étape 8 — Tests

**Lancer tous les tests :**
```bash
pytest test/ -v
```

**Ou test par test :**
```bash
pytest test/test_navigation.py -v        # Endpoints API
pytest test/test_fuzz_chaos.py -v        # Chaos / Fuzz / Stress
pytest test/test_charge_stabilite.py -v  # Charge et stabilité
pytest test/test_business_logic.py -v    # Cohérence des prédictions
pytest test/test_performances.py -v      # Comparaison prod vs v2
pytest test/test_monitoring_prod.py -v   # Monitoring post-déploiement
```

**Résultats actuels (production Render) :**

| Test | Statut |
|---|---|
| Navigation | ✅ SUCCESS |
| Chaos / Fuzz | ✅ SUCCESS |
| Charge / Stabilité | ✅ SUCCESS |
| Business Logic | ✅ SUCCESS |
| Monitoring prod | ✅ SUCCESS |

---

## 📊 Performances en production

| Métrique | Valeur |
|---|---|
| R² (test) | 0.978 |
| MAPE (test) | ~2% |
| R² (prod Render) | 0.950 |
| MAPE (prod Render) | 2.70% |
| Verdict monitoring | ✅ PROD STABLE |

---

## 🛠️ Installation locale

```bash
# Cloner le repo
git clone https://github.com/urbain-sarr1/Prediction_Consomation_Energie.git
cd Prediction_Consomation_Energie

# Installer les dépendances
pip install -r requirements.txt

# Préparer le dataset
python src/prepare_dataset.py

# Lancer l'API
uvicorn src.api:app --reload
```

---

## 📦 Dépendances principales

| Package | Version |
|---|---|
| fastapi | 0.115.6 |
| uvicorn | 0.34.0 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |
| joblib | 1.4.2 |
| holidays | 0.64 |
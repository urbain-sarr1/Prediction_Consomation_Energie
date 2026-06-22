# Prédiction de la consommation électrique journalière (France)

Modèle de Machine Learning qui prédit la consommation électrique journalière de
la France (en MW), à partir de la consommation passée, d'une prévision météo et
d'informations calendaires. Modèle de production : **XGBoost** (~1,9 % de MAPE,
sans utiliser la prévision de RTE).

## Structure du projet

```
src/
├── prepare_dataset.py          # construit le dataset enrichi (données RTE + météo + Tempo)
├── dataset_instinct.py         # sous-ensemble de variables choisi par hypothèse métier
├── feature_importance.py       # classement des variables (avec / sans prévision RTE)
├── train_models.py             # comparaison de 6 modèles (V1 / V2)
├── hyperparameter_tuning.py    # optimisation + sauvegarde du modèle de production
├── api.py                      # API de prédiction (FastAPI)
├── app_streamlit.py            # interface de démonstration (Streamlit)
├── test_api.py                 # script de test de l'API
├── requirements.txt
├── Dockerfile
└── output/
    ├── dataset_journalier_enrichi.csv
    └── modele_v2_optimise.joblib   # modèle entraîné (model + features + target)
```

## Variables utilisées par le modèle (9)

`Conso_J1`, `Conso_J7`, `Conso_moy_7j`, `Temperature`, `Temperature_max`,
`Jour_semaine_sin`, `Jour_semaine_cos`, `Est_weekend`, `Est_ferie`.

## Lancement en local

### Installer les dépendances
```bash
pip install -r requirements.txt
```

### Option A — API (FastAPI)
```bash
uvicorn api:app --reload
```
- Page de démonstration : http://127.0.0.1:8000/
- Prédiction : `POST http://127.0.0.1:8000/predict`
- Test rapide : `python test_api.py`

### Option B — Interface Streamlit
```bash
streamlit run app_streamlit.py
```
Ouvre automatiquement une page web avec un formulaire de prédiction.

## Déploiement avec Docker

### Construire l'image
```bash
docker build -t conso-elec .
```

### Lancer le conteneur (API)
```bash
docker run -p 8000:8000 conso-elec
```
L'API est alors accessible sur http://127.0.0.1:8000/

> Pour conteneuriser l'interface Streamlit à la place, voir la variante
> commentée à la fin du `Dockerfile`, puis exposer le port 8501.

## Ré-entraîner le modèle

Installer aussi `requests` et `matplotlib`, puis exécuter dans l'ordre :
```bash
python prepare_dataset.py        # 1. régénère output/dataset_journalier_enrichi.csv
python feature_importance.py     # 2. (optionnel) classement des variables
python train_models.py           # 3. comparaison des modèles
python hyperparameter_tuning.py  # 4. optimisation + nouveau modele_v2_optimise.joblib
```

## Maintenabilité

- **Versions** : les versions des bibliothèques sont fixées dans `requirements.txt`
  pour garantir des résultats reproductibles. Régénérer si besoin avec
  `pip freeze > requirements.txt` depuis l'environnement d'entraînement.
- **Ré-entraînement** : relancer le pipeline ci-dessus quand de nouvelles données
  RTE sont disponibles. Le modèle est sauvegardé avec la liste exacte de ses
  variables (`features`), ce qui évite toute incohérence côté API.
- **Surveillance** : suivre la MAPE sur les nouvelles données ; une dérive
  durable au-dessus de ~2,5 % indique qu'un ré-entraînement est nécessaire.
- **Données disponibles à la prédiction** : le modèle n'utilise que des variables
  connues la veille (consommation passée, prévision météo, calendrier), donc
  aucune fuite de données.

"""
API de prédiction de la consommation électrique journalière (France).

Charge le modèle entraîné (modele_prod.joblib), reçoit les informations
d'un jour à prédire, reconstruit les variables attendues par le modèle, et
renvoie la consommation prévue (en MW).

Lancer en local :
    uvicorn api:app --reload
Puis ouvrir la page de démonstration : http://127.0.0.1:8000/
"""

import os
import numpy as np
import pandas as pd
import joblib
import holidays
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# ------------------------------------------------------------
# Chargement du modèle (le .joblib contient : model + features + target)
# ------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "output", "model_prod.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_PATH}. "
        "Lance d'abord hyperparameter_tuning.py pour générer le .joblib."
    )

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["features"]          # ordre exact des variables attendues
TARGET = bundle.get("target", "Consommation")

fr_holidays = holidays.France()        # pour calculer Est_ferie à partir de la date

# ------------------------------------------------------------
# Application
# (docs auto désactivée : incompatibilité Pydantic/Python 3.9 -> on sert
#  notre propre page de démonstration à la place)
# ------------------------------------------------------------
app = FastAPI(
    title="Prédiction de consommation électrique",
    version="1.0.0",
    docs_url=None, redoc_url=None, openapi_url=None,
)


# ------------------------------------------------------------
# Schéma d'entrée
# ------------------------------------------------------------
class Entree(BaseModel):
    date: str = Field(..., description="Jour à prédire (AAAA-MM-JJ)")
    conso_j1: float = Field(..., description="Consommation de la veille (MW)")
    conso_j7: float = Field(..., description="Consommation il y a 7 jours (MW)")
    conso_moy_7j: float = Field(..., description="Moyenne des 7 derniers jours (MW)")
    temperature: float = Field(..., description="Température moyenne prévue du jour (°C)")
    temperature_max: float = Field(..., description="Température maximale prévue du jour (°C)")


# ------------------------------------------------------------
# Reconstruction des variables attendues par le modèle
# (mêmes calculs que dans prepare_dataset.py)
# ------------------------------------------------------------
def construire_features(e: Entree) -> pd.DataFrame:
    try:
        d = pd.to_datetime(e.date, format="%Y-%m-%d", errors="raise")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Date invalide. Format attendu : AAAA-MM-JJ."
        )

    jsem = d.dayofweek  # lundi = 0 ... dimanche = 6

    valeurs = {
        "Conso_J1": e.conso_j1,
        "Conso_J7": e.conso_j7,
        "Conso_moy_7j": e.conso_moy_7j,
        "Temperature": e.temperature,
        "Temperature_max": e.temperature_max,
        "Jour_semaine_sin": np.sin(2 * np.pi * jsem / 7),
        "Jour_semaine_cos": np.cos(2 * np.pi * jsem / 7),
        "Est_weekend": 1 if jsem >= 5 else 0,
        "Est_ferie": 1 if d.date() in fr_holidays else 0,
    }

    manquantes = [f for f in FEATURES if f not in valeurs]
    if manquantes:
        raise HTTPException(
            status_code=500,
            detail=f"Variables non calculées par l'API : {manquantes}"
        )

    return pd.DataFrame([[valeurs[f] for f in FEATURES]], columns=FEATURES)


# ------------------------------------------------------------
# Page de démonstration (remplace /docs)
# ------------------------------------------------------------
PAGE_HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Prédiction de consommation électrique</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 540px; margin: 40px auto;
         color: #1f1f1f; padding: 0 16px; }
  h1 { font-size: 22px; border-bottom: 2px solid #2E5E8C; padding-bottom: 8px; color: #2E5E8C; }
  label { display: block; margin: 14px 0 4px; font-size: 14px; font-weight: bold; }
  input { width: 100%; padding: 8px; font-size: 15px; box-sizing: border-box;
          border: 1px solid #ccc; border-radius: 4px; }
  button { margin-top: 20px; width: 100%; padding: 12px; font-size: 16px; font-weight: bold;
           background: #2E5E8C; color: #fff; border: none; border-radius: 4px; cursor: pointer; }
  button:hover { background: #244c73; }
  #resultat { margin-top: 22px; padding: 16px; border-radius: 6px; font-size: 18px;
              text-align: center; display: none; }
  .ok { background: #e8f1ea; color: #2b6b46; }
  .err { background: #fdecea; color: #b3261e; }
  .hint { font-size: 12px; color: #777; font-weight: normal; }
</style>
</head>
<body>
  <h1>Prédiction de consommation électrique</h1>
  <p class="hint">Renseigne un jour à prédire, la consommation passée et la prévision météo.</p>

  <label>Date à prédire</label>
  <input type="date" id="date" value="2026-01-15">

  <label>Consommation de la veille (MW)</label>
  <input type="number" id="conso_j1" value="62000">

  <label>Consommation il y a 7 jours (MW)</label>
  <input type="number" id="conso_j7" value="61000">

  <label>Moyenne des 7 derniers jours (MW)</label>
  <input type="number" id="conso_moy_7j" value="60500">

  <label>Température moyenne prévue (°C)</label>
  <input type="number" step="0.1" id="temperature" value="4.5">

  <label>Température maximale prévue (°C)</label>
  <input type="number" step="0.1" id="temperature_max" value="8.0">

  <button onclick="predire()">Prédire la consommation</button>
  <div id="resultat"></div>

<script>
async function predire() {
  const corps = {
    date: document.getElementById('date').value,
    conso_j1: parseFloat(document.getElementById('conso_j1').value),
    conso_j7: parseFloat(document.getElementById('conso_j7').value),
    conso_moy_7j: parseFloat(document.getElementById('conso_moy_7j').value),
    temperature: parseFloat(document.getElementById('temperature').value),
    temperature_max: parseFloat(document.getElementById('temperature_max').value),
  };
  const box = document.getElementById('resultat');
  box.style.display = 'block';
  box.className = '';
  box.textContent = 'Calcul en cours...';
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(corps),
    });
    const data = await r.json();
    if (r.ok) {
      box.className = 'ok';
      box.innerHTML = 'Consommation prévue le ' + data.date + ' :<br><b>' +
                      Math.round(data.consommation_prevue_MW).toLocaleString('fr-FR') + ' MW</b>';
    } else {
      box.className = 'err';
      box.textContent = 'Erreur : ' + (data.detail || JSON.stringify(data));
    }
  } catch (e) {
    box.className = 'err';
    box.textContent = 'Erreur de connexion à l\\'API.';
  }
}
</script>
</body>
</html>
"""


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def page_demo():
    return PAGE_HTML


@app.get("/health")
def health():
    return {"status": "ok", "modele_charge": True, "nb_features": len(FEATURES)}


@app.post("/predict")
def predict(e: Entree):
    X = construire_features(e)
    prevision = float(model.predict(X)[0])
    return {"date": e.date, "consommation_prevue_MW": round(prevision, 1)}


# ------------------------------------------------------------
# Lancement direct : python api.py
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
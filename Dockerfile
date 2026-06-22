# ============================================================
# DOCKERFILE
# Image de l'API FastAPI de prédiction de consommation électrique
# ============================================================

# Base Python 3.11 légère
FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie et installation des dépendances en premier
# (optimisation cache Docker : réinstallation seulement si requirements.txt change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source uniquement
# Les modèles (src/output/*.joblib) sont montés via volume dans docker-compose
# → pas besoin de les copier ici, ils seront toujours à jour après un swap
COPY src ./src

# Port d'écoute de l'API
EXPOSE 8000

# Lancement de l'API FastAPI avec Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
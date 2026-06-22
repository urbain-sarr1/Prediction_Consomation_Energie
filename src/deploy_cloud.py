import os
import subprocess
import sys

# ============================================================
# DEPLOY.PY
# Equivalent cloud de : docker-compose down + build + up -d
# Usage : python deploy.py "message du commit"
# ============================================================

def run(cmd: str):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Erreur sur : {cmd}")
        sys.exit(1)

def deploy(message: str = "new model prod"):

    print("\n" + "=" * 60)
    print("🚀 DÉPLOIEMENT CLOUD")
    print("=" * 60)

    # Vérifie que model_prod.joblib existe
    model_path = os.path.join("src", "output", "model_prod.joblib")
    if not os.path.exists(model_path):
        print(f"❌ model_prod.joblib introuvable : {model_path}")
        sys.exit(1)

    print(f"✅ model_prod.joblib trouvé")

    # Equivalent docker-compose down → rien à faire, Render gère
    # Equivalent docker-compose build → git push déclenche le rebuild sur Render
    # Equivalent docker-compose up    → Render redémarre automatiquement

    run("git add src/output/model_prod.joblib")
    run(f'git commit -m "{message}"')
    run("git push origin main")

    print("\n" + "=" * 60)
    print("✅ DÉPLOIEMENT LANCÉ")
    print("Render rebuild et redémarre l'API automatiquement.")
    print("Suivi : https://dashboard.render.com")
    print("=" * 60)

if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "new model prod"
    deploy(message)
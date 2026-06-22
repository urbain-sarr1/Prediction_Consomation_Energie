import os
import sys
import shutil
from datetime import datetime

# ============================================================
# FIX IMPORT PATH (IMPORTANT)
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ============================================================
# IMPORT MODULES
# ============================================================

from src.optimisation_hyperparametres import optimisation_hyperparametres
from test.test_performances import test_performance_api


# ============================================================
# SAFE MODEL SWAP (rotation circulaire à 3 sans perte)
# ============================================================

def safe_swap_models(model_dir: str):
    """
    Effectue le cycle de rotation des modèles de façon sûre :
        prod   -> backup
        v2     -> prod
        backup -> v2

    Principe : aucun fichier n'est jamais écrasé directement.
    On déplace chaque fichier vers un nom temporaire dédié avant
    de toucher sa destination finale, puis on renomme les
    temporaires vers leur place définitive. Ainsi, même si le
    script crash au milieu, aucun des trois fichiers n'est perdu :
    soit le swap est complet, soit rien n'a bougé de façon
    irréversible (les .tmp restent récupérables).
    """
    path_prod = os.path.join(model_dir, "model_prod.joblib")
    path_v2 = os.path.join(model_dir, "model_v2.joblib")
    path_backup = os.path.join(model_dir, "model_backup.joblib")

    # Noms temporaires, un par fichier en mouvement
    tmp_prod = os.path.join(model_dir, "model_prod.joblib.tmp")
    tmp_v2 = os.path.join(model_dir, "model_v2.joblib.tmp")
    tmp_backup = os.path.join(model_dir, "model_backup.joblib.tmp")

    if not os.path.exists(path_v2):
        raise FileNotFoundError(f"model_v2 introuvable, swap annulé : {path_v2}")

    # 0. Filet de sécurité supplémentaire : copie horodatée de l'ancien prod
    if os.path.exists(path_prod):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety_copy = os.path.join(model_dir, f"model_prod_{timestamp}.joblib")
        shutil.copy2(path_prod, safety_copy)
        print(f"🛟 Copie de sécurité créée : {safety_copy}")

    # ------------------------------------------------------
    # PHASE 1 : on sort les 3 fichiers de leur emplacement
    #           actuel vers un .tmp — rien n'est écrasé ici
    # ------------------------------------------------------
    if os.path.exists(path_prod):
        os.replace(path_prod, tmp_prod)      # ancien prod   -> tmp_prod
    if os.path.exists(path_v2):
        os.replace(path_v2, tmp_v2)          # nouveau (v2)  -> tmp_v2
    if os.path.exists(path_backup):
        os.replace(path_backup, tmp_backup)  # ancien backup -> tmp_backup

    # ------------------------------------------------------
    # PHASE 2 : on range chaque .tmp à sa destination finale
    #           selon le cycle prod->backup, v2->prod, backup->v2
    # ------------------------------------------------------
    if os.path.exists(tmp_prod):
        os.replace(tmp_prod, path_backup)    # ancien prod   -> backup
    if os.path.exists(tmp_v2):
        os.replace(tmp_v2, path_prod)        # nouveau (v2)  -> prod
    if os.path.exists(tmp_backup):
        os.replace(tmp_backup, path_v2)      # ancien backup -> v2

    print("🔄 MODEL SWAP COMPLETED")
    print("   - model_prod (ancien) → model_backup")
    print("   - model_v2 (nouveau)  → model_prod")
    print("   - model_backup (ancien) → model_v2")


# ============================================================
# PIPELINE RETRAIN
# ============================================================

def run_retrain_pipeline():

    print("\n" + "=" * 60)
    print("🚀 START RETRAIN PIPELINE")
    print("=" * 60)

    # ========================================================
    # STEP 1 : OPTIMISATION HYPERPARAMÈTRES
    # ========================================================

    print("\n🔧 Step 1 : Hyperparameter optimization...")
    optimisation_hyperparametres()
    print("✅ Optimization finished (model_v2 + results CSV generated)")

    # ========================================================
    # STEP 2 : TEST PERFORMANCE (PROD vs V2)
    # ========================================================

    print("\n🧪 Step 2 : Performance comparison...")
    result = test_performance_api(mode="retrain")

    # ========================================================
    # STEP 3 : RESULT
    # ========================================================

    print("\n📊 Step 3 : RESULT")

    if result == 1:
        print("❌ New model is NOT better than production")
        print("👉 CAN'T DEPLOY model_v2")

    else:
        print("✅ New model is better than production")
        print("👉 CAN DEPLOY model_v2")

        model_dir = os.path.join(ROOT_DIR, "src", "output")

        try:
            safe_swap_models(model_dir)
            print("\n🚀 DEPLOY FINISH")
        except Exception as e:
            print(f"\n⛔ DEPLOY ABORTED — swap error: {e}")
            print("👉 vérifier l'état du dossier output (fichiers .tmp éventuels)")

    print("\n" + "=" * 60)
    print("🏁 PIPELINE FINISHED")
    print("=" * 60)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    run_retrain_pipeline()
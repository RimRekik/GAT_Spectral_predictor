import os
import re
import glob

logs = glob.glob("logs/train_run_*.log")
results = []

for log_path in logs:
    with open(log_path, "r") as f:
        content = f.read()
        
        # Récupération des paramètres affichés dans l'écho du script
        param_match = re.search(r"Paramètres : LR=(.*?), DIM=(.*?), LAYERS=(.*)", content)
        losses = re.findall(r"Val Loss:\s*([0-9.]+)", content)
        
        if param_match and losses:
            lr, dim, layers = param_match.groups()
            best_loss = min(float(l) for l in losses)
            results.append({
                "lr": lr, "dim": dim, "layers": layers, "val_loss": best_loss
            })

# Tri et affichage du tableau récapitulatif
results.sort(key=lambda x: x["val_loss"])
print("\n=== CLASSEMENT DES MEILLEURS MODÈLES ===")
for rank, res in enumerate(results, 1):
    print(f"Top {rank} | Val Loss: {res['val_loss']:.4f} | LR: {res['lr']}, DIM: {res['dim']}, Layers: {res['layers']}")
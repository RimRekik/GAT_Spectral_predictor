import subprocess
import itertools
import os
import re

# 1. Définition de ta grille (18 combinaisons)
grid = {
    "lr": [0.001, 0.0001],
    "batch_size": [256],
    "hidden_dim": [64, 128, 256],
    "num_layers": [3, 5, 7]
}

keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Dictionnaire pour stocker les scores de chaque run
resultats_globaux = []

print(f"=== Début du Grid Search : {len(experiments)} configurations ===")

for i, params in enumerate(experiments, 1):
    print(f"\n[Run {i}/{len(experiments)}] Configuration : {params}")
    
    nom_modele = f"model_lr{params['lr']}_dim{params['hidden_dim']}_layers{params['num_layers']}.pt"
    save_path = f"model/{nom_modele}"
    
    # On crée un fichier de log temporaire pour ce run précis
    log_file = f"run_{i}_output.txt"
    
    cmd = [
        "python", "../main.py",
        "--lr", str(params["lr"]),
        "--batch_size", str(params["batch_size"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--num_layers", str(params["num_layers"]),
        "--save_path", save_path,
        "--max_steps", "5000"
    ]
    
    try:
        # On exécute en affichant dans le terminal ET en enregistrant dans le fichier log_file
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="") # Affiche la barre de progression en direct
                f.write(line)
            process.wait()
        
        # --- Extraction de la meilleure Val Loss ---
        best_val_loss = "Inconnue (Erreur de lecture)"
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                # On cherche les lignes du type : "Val Loss: 0.8192"
                losses = re.findall(r"Val Loss:\s*([0-9.]+)", content)
                if losses:
                    # On prend la plus petite valeur trouvée dans ce run
                    best_val_loss = min(float(l) for l in losses)
            
            # Nettoyage du fichier log temporaire
            os.remove(log_file)

        print(f"-> RUN {i} TERMINÉ. Meilleure Val Loss détectée : {best_val_loss}")
        
        # Enregistrement du résultat
        resultats_globaux.append({
            "run": i,
            "params": params,
            "val_loss": best_val_loss
        })

    except Exception as e:
        print(f"-> /!\\ Le RUN {i} a échoué. Erreur : {e}")
        resultats_globaux.append({
            "run": i,
            "params": params,
            "val_loss": "ÉCHEC"
        })

# --- ÉTAPE FINALE : LE TABLEAU RÉCAPITULATIF ---
print("\n" + "="*50)
print("         RÉSULTATS DU GRID SEARCH")
print("="*50)

# Trier les résultats du meilleur au moins bon (exclure les échecs du tri)
resultats_globaux.sort(key=lambda x: x["val_loss"] if isinstance(x["val_loss"], float) else float('inf'))

# Affichage dans le terminal et sauvegarde dans un fichier texte
with open("rapport_grid_search.txt", "w") as f_rapport:
    f_rapport.write("=== CLASSEMENT DES MEILLEURS MODÈLES (Plus petite Val Loss en premier) ===\n\n")
    
    for res in resultats_globaux:
        ligne = f"Top {res['run']} | Val Loss: {res['val_loss']} | Paramètres: {res['params']}\n"
        print(ligne, end="")
        f_rapport.write(ligne)

print("\n" + "="*50)
print("Le classement a été sauvegardé dans : rapport_grid_search.txt")
print("="*50)
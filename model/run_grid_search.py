import subprocess
import itertools

# Définition des paramètres à tester (2 x 2 x 2 = 8 runs)
grid = {
    "lr": [0.001, 0.0005],
    "batch_size": [256, 512],
    "hidden_dim": [128, 256]
}

# Génère automatiquement toutes les combinaisons possibles
keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"=== Lancement du Grid Search ({len(experiments)} combinaisons) ===")

for i, params in enumerate(experiments, 1):
    print(f"\n[RUN {i}/{len(experiments)}] Paramètres : {params}")
    
    # On donne un nom unique au fichier sauvegardé pour ne pas écraser le précédent
    nom_modele = f"model_lr{params['lr']}_bs{params['batch_size']}_dim{params['hidden_dim']}.pt"
    save_path = f"model/{nom_modele}"
    
    # On construit la commande pour exécuter main.py avec les bonnes options
    cmd = [
        "python", "main.py",
        "--lr", str(params["lr"]),
        "--batch_size", str(params["batch_size"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--save_path", save_path,
        "--max_steps", "5000"  # 5000 étapes par run pour aller plus vite (ajustable)
    ]
    
    # Exécution du script principal
    try:
        subprocess.run(cmd, check=True)
        print(f"-> RUN {i} REUSSI. Modèle sauvegardé sous : {save_path}")
    except subprocess.CalledProcessError as e:
        print(f"-> /!\\ Le RUN {i} a échoué. Erreur : {e}")

print("\n=== Grid Search terminé avec succès ! ===")
import subprocess
import itertools

# Grille optimisée (18 runs)
grid = {
    "lr": [0.001, 0.0001],
    "batch_size": [256],
    "hidden_dim": [64, 128, 256],
    "num_layers": [3, 5, 7]
}

keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"=== Début du Grid Search : {len(experiments)} configurations ===")

for i, params in enumerate(experiments, 1):
    print(f"\n[Run {i}/{len(experiments)}] Configuration : {params}")
    
    nom_modele = f"model_lr{params['lr']}_dim{params['hidden_dim']}_layers{params['num_layers']}.pt"
    save_path = f"model/{nom_modele}"
    
    cmd = [
        "python", "../main.py",
        "--lr", str(params["lr"]),
        "--batch_size", str(params["batch_size"]),
        "--hidden_dim", str(params["hidden_dim"]),
        "--num_layers", str(params["num_layers"]), # <-- TRÈS IMPORTANT de rajouter cette ligne !
        "--save_path", save_path,
        "--max_steps", "5000"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur run {i} : {e}")
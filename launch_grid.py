import os
import subprocess
import itertools

grid = {
    "lr": [0.001, 0.0001,0.00001],
    "batch_size": [256],
    "hidden_dim": [128, 256],
    "num_layers": [3, 5, 7]
}

keys, values = zip(*grid.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

# S'assurer que les dossiers nécessaires existent
os.makedirs("logs", exist_ok=True)
os.makedirs("model", exist_ok=True)
os.makedirs("submit_scripts", exist_ok=True)

# Lire le fichier template Slurm
with open("sbatch_template.sh", "r") as f:
    template_content = f.read()

print(f"=== Soumission de {len(experiments)} jobs Slurm en parallèle ===")

for i, params in enumerate(experiments, 1):
    nom_modele = f"model_lr{params['lr']}_dim{params['hidden_dim']}_layers{params['num_layers']}.pt"
    save_path = f"model/{nom_modele}"
    
    # Remplacement des placeholders du template par les vraies valeurs du run
    job_content = template_content
    job_content = job_content.replace("__LR__", str(params["lr"]))
    job_content = job_content.replace("__BATCH_SIZE__", str(params["batch_size"]))
    job_content = job_content.replace("__DIM__", str(params["hidden_dim"]))
    job_content = job_content.replace("__LAYERS__", str(params["num_layers"]))
    job_content = job_content.replace("__SAVE_PATH__", save_path)
    
    # Fichier de soumission temporaire propre à ce run
    submit_file = f"submit_scripts/run_{i}.sh"
    with open(submit_file, "w") as f_out:
        f_out.write(job_content)
        
    # Soumission effective à Slurm
    print(f"[Run {i}/{len(experiments)}] Soumission -> LR: {params['lr']}, DIM: {params['hidden_dim']}, LAYERS: {params['num_layers']}")
    subprocess.run(["sbatch", submit_file])

print("\nTous les jobs ont été soumis. Utilisez 'squeue -u rrekikdi' pour suivre l'avancement.")
"""
Script de diagnostic : compare la taille annoncée dans meta.txt avec la
taille réelle (nombre de graphes) de chaque chunk .pt, pour confirmer
l'hypothèse du bug "IndexError: list index out of range" dans
HierarchicalStreamingSpectraDataset.get().

Usage:
    python diagnose_chunks.py /chemin/vers/root_train
    python diagnose_chunks.py /chemin/vers/root_val
    python diagnose_chunks.py /chemin/vers/root_test
"""

import os
import sys
import torch


def diagnose(root):
    meta_file = os.path.join(root, "meta.txt")

    if not os.path.exists(meta_file):
        print(f"[ERREUR] meta.txt introuvable dans {root}")
        return

    print(f"\n{'='*70}")
    print(f"Diagnostic pour : {root}")
    print(f"{'='*70}")

    chunk_files = []
    declared_sizes = []

    with open(meta_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, size = line.strip().split(",")
            chunk_files.append(path)
            declared_sizes.append(int(size))

    total_declared = sum(declared_sizes)
    total_real = 0
    mismatches = []

    print(f"Nombre de chunks listés dans meta.txt : {len(chunk_files)}")
    print(f"Total annoncé (somme des sizes)        : {total_declared}\n")

    for i, (path, declared) in enumerate(zip(chunk_files, declared_sizes)):
        if not os.path.exists(path):
            print(f"[CHUNK {i}] MANQUANT sur disque : {path}")
            mismatches.append((i, path, declared, None))
            continue

        try:
            data_list = torch.load(path, weights_only=False)
            real = len(data_list)
        except Exception as e:
            print(f"[CHUNK {i}] ERREUR de chargement {path} : {e}")
            mismatches.append((i, path, declared, "load_error"))
            continue

        total_real += real

        if real != declared:
            status = "MANQUE" if real < declared else "EXCEDENT"
            print(
                f"[CHUNK {i}] {status} -- annoncé={declared:6d}  "
                f"réel={real:6d}  écart={real - declared:+d}   ({path})"
            )
            mismatches.append((i, path, declared, real))

    print(f"\n{'-'*70}")
    print(f"Total réel (graphes effectivement chargeables) : {total_real}")
    print(f"Total annoncé (meta.txt)                        : {total_declared}")
    print(f"Écart total                                      : {total_real - total_declared:+d}")

    if mismatches:
        print(f"\n⚠️  {len(mismatches)} chunk(s) avec un écart détecté sur {len(chunk_files)}.")
        print("    -> C'est la cause probable de l'IndexError dans dataset.get():")
        print("       cumulative_sizes utilise les tailles ANNONCÉES, donc des")
        print("       indices valides selon meta.txt tombent hors de la vraie liste.")
    else:
        print("\n✅ Aucun écart détecté : meta.txt est cohérent avec le contenu réel.")

    return mismatches


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_chunks.py <root_dir> [<root_dir2> ...]")
        sys.exit(1)

    all_mismatches = {}
    for root in sys.argv[1:]:
        all_mismatches[root] = diagnose(root)

    print(f"\n{'='*70}")
    print("RÉSUMÉ GLOBAL")
    print(f"{'='*70}")
    for root, mismatches in all_mismatches.items():
        n = len(mismatches) if mismatches else 0
        flag = "⚠️  PROBLÈME" if n > 0 else "✅ OK"
        print(f"{flag} -- {root} -- {n} chunk(s) en écart")
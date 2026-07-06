import argparse
from model.model import MODEL_REGISTRY


def load_args():
    parser = argparse.ArgumentParser()

    # ── Training ──────────────────────────────────────────
    parser.add_argument('--max_steps',  type=int,   default=10000)
    parser.add_argument('--eval_every', type=int,   default=300)
    parser.add_argument('--eval_inter', type=int,   default=1)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int,   default=512)
    parser.add_argument('--save_path',  type=str,   default='model/model.pt')

    # ── Data ──────────────────────────────────────────────
    parser.add_argument('--root_train', type=str,
        default='/store/pmcs2i/AIDIBOP/hierarchical_dataset/hierarchical_dataset/processed_graphs_train_hcd_hierarchical_dummy')
    parser.add_argument('--root_val',   type=str,
        default='/store/pmcs2i/AIDIBOP/hierarchical_dataset/hierarchical_dataset/processed_graphs_val_hcd_hierarchical_dummy')
    parser.add_argument('--root_test',  type=str,
        default='/store/pmcs2i/AIDIBOP/hierarchical_dataset/hierarchical_dataset/processed_graphs_holdout_hcd_hierarchical_dummy')

    # ── Model selection ───────────────────────────────────
    parser.add_argument('--model', type=str, default='cyclic_gat_global',
                        choices=list(MODEL_REGISTRY),
                        help=f'Modèle à utiliser. Choix : {list(MODEL_REGISTRY)}')

    # ── Model hyperparameters ─────────────────────────────
    parser.add_argument('--hidden_dim',     type=int,   default=256)
    parser.add_argument('--num_layers',     type=int,   default=3)
    parser.add_argument('--heads',          type=int,   default=4)
    parser.add_argument('--dropout',        type=float, default=0.2)
    parser.add_argument('--num_timesteps',  type=int,   default=2,
                        help='Utilisé uniquement par AttentiveFP')
    parser.add_argument('--max_aa_aa_edges', type=int,  default=None,
                        help='Utilisé uniquement par les modèles avec readout par arête')

    return parser.parse_args()
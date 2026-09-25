import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_steps', type=int, default=2500)
    parser.add_argument('--eval_every', type=int, default=100)
    parser.add_argument('--eval_inter', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--save_path', type=str, default='model/model.pt')
    parser.add_argument('--root_train', type=str, default='/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/processed_dataset/hierarchical_dataset/processed_graphs_train_hcd_hierarchical_dummy')
    parser.add_argument('--root_val', type=str, default='/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/processed_dataset/hierarchical_dataset/processed_graphs_val_hcd_hierarchical_dummy')
    parser.add_argument('--root_test', type=str, default='/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/processed_dataset/hierarchical_dataset/processed_graphs_holdout_hcd_hierarchical_dummy')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--model_type', type=str, default='GAT')
    parser.add_argument('--graph_structure', type=str, default='complete',
                         choices=['complete', 'atom_aa', 'aa_only', 'atomic_only'],
                         help="Graph-structure ablation condition. All four conditions "
                              "('complete', 'atom_aa', 'aa_only', 'atomic_only') prune the same "
                              "hierarchical dataset (root_* must point to a hierarchical dataset) "
                              "down to the requested node/edge subset before training a "
                              "hierarchical model unchanged. 'atomic_only' keeps atom nodes and "
                              "atom-atom edges only.")
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_timesteps', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--load_weights', type=str, default=None)
    parser.add_argument('--jumping_knowledge', action='store_true', default=False,
                         help="If set, the readout pools a projection of ALL intermediate "
                              "layer states (concatenated) instead of only the last layer's "
                              "output. Mitigates over-smoothing on small node sets (e.g. "
                              "aa_only's 19 AA nodes with many message-passing layers). "
                              "Off by default to not alter existing runs/results.")
    args = parser.parse_args()

    return args


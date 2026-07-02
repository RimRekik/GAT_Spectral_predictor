"""
verification for the hierarchical peptide graph pipeline.

"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_undirected

NODE_TYPE_DIM = 3   # [atom, aa, global]
EDGE_TYPE_DIM = 4   # [atom_atom, atom_aa, aa_aa, aa_global]


# --------------------------------------------------------------------------
# 1) Reproduce process_one()'s index bookkeeping for a synthetic "molecule"
# --------------------------------------------------------------------------
def build_synthetic_graph(total_atom, total_aa, feat_dim=8, bond_dim=5, seed=0):
    
    rng = np.random.default_rng(seed)
    global_idx = total_atom + total_aa 

    edges_atom_atom = [(i, i + 1) for i in range(total_atom - 1)]

    atoms_per_aa = np.array_split(np.arange(total_atom), total_aa)
    edges_atom_aa = []
    for aa_i, atom_idxs in enumerate(atoms_per_aa):
        for a in atom_idxs:
            edges_atom_aa.append((int(a), total_atom + aa_i))

    edges_aa_aa = [(total_atom + idx, total_atom + idx + 1) for idx in range(total_aa - 1)]

    edges_aa_global = [(total_atom + idx, global_idx) for idx in range(total_aa)]

    def arr(e):
        return np.asarray(e, dtype=np.int64) if len(e) else np.empty((0, 2), dtype=np.int64)

    edges = np.concatenate([arr(edges_atom_atom), arr(edges_atom_aa),
                             arr(edges_aa_aa), arr(edges_aa_global)], axis=0)
    edge_index = edges.T

    edge_attr_atom_atom = np.concatenate([
        rng.normal(size=(len(edges_atom_atom), bond_dim)),
        np.tile([1, 0, 0, 0], (len(edges_atom_atom), 1))
    ], axis=1)
    edge_attr_atom_aa = np.concatenate([
        np.zeros((len(edges_atom_aa), bond_dim)),
        np.tile([0, 1, 0, 0], (len(edges_atom_aa), 1))
    ], axis=1)
    edge_attr_aa_aa = np.concatenate([
        np.zeros((len(edges_aa_aa), bond_dim)),
        np.tile([0, 0, 1, 0], (len(edges_aa_aa), 1))
    ], axis=1)
    edge_attr_aa_global = np.concatenate([
        np.zeros((len(edges_aa_global), bond_dim)),
        np.tile([0, 0, 0, 1], (len(edges_aa_global), 1))
    ], axis=1)

    edge_attr = np.concatenate(
        [edge_attr_atom_atom, edge_attr_atom_aa, edge_attr_aa_aa, edge_attr_aa_global], axis=0
    )

    num_nodes = total_atom + total_aa + 1
    node_type = np.concatenate([
        np.tile([1, 0, 0], (total_atom, 1)),
        np.tile([0, 1, 0], (total_aa, 1)),
        np.tile([0, 0, 1], (1, 1)),
    ], axis=0).astype(np.float32)
    x_feat = rng.normal(size=(num_nodes, feat_dim)).astype(np.float32)
    x = np.concatenate([x_feat, node_type], axis=1)

    aa_bond_idx_of_node = -np.ones(num_nodes, dtype=np.int64)
    for idx in range(total_aa):
        aa_bond_idx_of_node[total_atom + idx] = idx  # node's residue rank

    return {
        "x": torch.from_numpy(x).float(),
        "edge_index": torch.from_numpy(edge_index).long(),
        "edge_attr": torch.from_numpy(edge_attr).float(),
        "y": torch.zeros(174),
        "total_atom": total_atom,
        "total_aa": total_aa,
        "global_idx": global_idx,
        "aa_bond_idx_of_node": torch.from_numpy(aa_bond_idx_of_node),
    }


def to_pyg_data(g):
    ei, ea = to_undirected(g["edge_index"], g["edge_attr"])
    data = Data(x=g["x"], edge_index=ei, edge_attr=ea, y=g["y"])
    data.aa_bond_idx_of_node = g["aa_bond_idx_of_node"]
    return data


# --------------------------------------------------------------------------
# 2) Reproduce the model's edge-type split + aa-aa readout-ordering logic
# --------------------------------------------------------------------------
def split_edges_by_type(edge_index, edge_attr, type_dim):
    type_onehot = edge_attr[:, -type_dim:]
    type_id = type_onehot.argmax(dim=1)
    out = {}
    for t in range(type_dim):
        mask = type_id == t
        out[t] = (edge_index[:, mask], edge_attr[mask])
    return out


def check_aa_aa_ordering(batch_data_list, verbose=True):
 
    batch = Batch.from_data_list(batch_data_list)
    edge_index, edge_attr, b = batch.edge_index, batch.edge_attr, batch.batch
    aa_bond_idx_of_node = batch.aa_bond_idx_of_node

    edges_by_type = split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)
    ei_aa_aa, _ = edges_by_type[2]

    keep = ei_aa_aa[0] < ei_aa_aa[1]
    src, dst = ei_aa_aa[0, keep], ei_aa_aa[1, keep]

    # "ground truth" residue-bond index for each surviving edge, read off
    # the node's own stored aa_bond_idx_of_node (independent of any sorting)
    true_idx = aa_bond_idx_of_node[src]

    graph_id = b[src]
    batch_size = int(b.max().item()) + 1
    local_rank = torch.zeros_like(graph_id)
    for g in range(batch_size):
        g_mask = graph_id == g
        n_edges_g = int(g_mask.sum().item())
        local_rank[g_mask] = torch.arange(n_edges_g)

    mismatches = (local_rank != true_idx).nonzero(as_tuple=True)[0]

    if verbose:
        print(f"  batch_size={batch_size}, total surviving aa-aa edges={src.numel()}")
        for g in range(batch_size):
            g_mask = (graph_id == g)
            print(f"    graph {g}: local_rank={local_rank[g_mask].tolist()} "
                  f"true_idx={true_idx[g_mask].tolist()}")
    if mismatches.numel() == 0:
        print("  RESULT: local_rank == true residue-bond idx for every edge. Scatter mapping is correct.")
    else:
        print(f"  RESULT: MISMATCH at {mismatches.numel()} edge(s)! local_rank does NOT reflect "
              f"true residue order here -> out[..] scatter would land in the wrong y slot.")
        for m in mismatches.tolist():
            print(f"    -> edge pos {m}: local_rank={local_rank[m].item()} vs true_idx={true_idx[m].item()}")
    return mismatches.numel() == 0


# --------------------------------------------------------------------------
# 3) Run checks across a range of peptide sizes, batch sizes, and graph orderings
# --------------------------------------------------------------------------
def describe_graph(g, name=""):
    total_atom, total_aa = g["total_atom"], g["total_aa"]
    n_nodes = total_atom + total_aa + 1
    n_edges_directed = g["edge_index"].shape[1]
    print(f"[{name}] nodes: atom={total_atom}, aa={total_aa}, global=1, total={n_nodes}")
    print(f"[{name}] directed edges (pre to_undirected): {n_edges_directed} "
          f"(atom_atom={total_atom-1}, atom_aa={total_atom}, aa_aa={total_aa-1}, aa_global={total_aa})")


# --------------------------------------------------------------------------
# 6) SAME check, but on REAL data loaded via your actual Dataset + DataLoader
# --------------------------------------------------------------------------
def check_real_dataset(dataset_root, num_batches=5, batch_size=8, num_workers=0):

    import sys
    sys.path.insert(0, dataset_root if dataset_root not in sys.path else ".")

    from torch_geometric.loader import DataLoader
    from data.hierarchical_streaming_dataset import HierarchicalStreamingSpectraDataset

    dataset = HierarchicalStreamingSpectraDataset(root=dataset_root)

    import os
    def resolve_chunk_path(raw_path, root):
        candidates = [
            raw_path,                                    # as stored, maybe already absolute
            os.path.join(root, raw_path),                # relative to root
            os.path.join(root, os.path.basename(raw_path)),  # just the filename, in root
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return raw_path  # nothing matched; keep original so the error message is informative

    dataset.chunk_files = [resolve_chunk_path(f, dataset_root) for f in dataset.chunk_files]
    missing = [f for f in dataset.chunk_files if not os.path.exists(f)]
    if missing:
        print(f"WARNING: {len(missing)} chunk file(s) referenced in meta.txt "
              f"could not be resolved to an existing file, e.g. {missing[0]}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    all_ok = True
    b_idx = -1
    for b_idx, batch in enumerate(loader):
        if b_idx >= num_batches:
            break

        edge_index, edge_attr, b = batch.edge_index, batch.edge_attr, batch.batch
        edges_by_type = split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)
        ei_aa_aa, _ = edges_by_type[2]

        keep = ei_aa_aa[0] < ei_aa_aa[1]
        src, dst = ei_aa_aa[0, keep], ei_aa_aa[1, keep]
        graph_id = b[src]

        batch_size_actual = int(b.max().item()) + 1
        batch_ok = True
        for g in range(batch_size_actual):
            g_mask = (graph_id == g)
            src_g = src[g_mask]
            if src_g.numel() == 0:
                continue
            is_sorted = torch.all(src_g[1:] > src_g[:-1]).item() if src_g.numel() > 1 else True
            # and must form a contiguous run, no gaps/duplicates skipped
            contiguous = torch.all((src_g[1:] - src_g[:-1]) == 1).item() if src_g.numel() > 1 else True
            if not (is_sorted and contiguous):
                batch_ok = False
                print(f"  [batch {b_idx}] graph {g}: ORDER PROBLEM -> "
                      f"src node ids = {src_g.tolist()} "
                      f"(sorted={is_sorted}, contiguous={contiguous})")

        n_edges_total = src.numel()
        n_graphs = batch_size_actual
        status = "OK" if batch_ok else "MISMATCH"
        print(f"[batch {b_idx}] graphs={n_graphs}, aa-aa edges kept={n_edges_total} -> {status}")
        all_ok = all_ok and batch_ok

    print(f"\nReal-data check over {b_idx + 1} batches: "
          f"{'ALL OK - local_rank ordering assumption holds' if all_ok else 'PROBLEM FOUND - see above'}")
    return all_ok


if __name__ == "__main__":
    print("=" * 70)
    print("STEP 1 - single graph, structural sanity check")
    print("=" * 70)
    g = build_synthetic_graph(total_atom=20, total_aa=5, seed=1)
    describe_graph(g, "peptide_len5")
    data = to_pyg_data(g)
    print(f"edge_index after to_undirected: {data.edge_index.shape[1]} edges "
          f"(should be 2x directed count, since none are self-loops/duplicates)")

    print()
    print("=" * 70)
    print("STEP 2 - aa-aa ordering check, SINGLE graph in a batch of 1")
    print("=" * 70)
    check_aa_aa_ordering([data])

    print()
    print("=" * 70)
    print("STEP 3 - aa-aa ordering check, BATCH of graphs with DIFFERENT peptide lengths")
    print("=" * 70)
    graphs = []
    for i, (ta, taa) in enumerate([(20, 5), (8, 2), (35, 9), (14, 4)]):
        gi = build_synthetic_graph(total_atom=ta, total_aa=taa, seed=10 + i)
        describe_graph(gi, f"graph{i}")
        graphs.append(to_pyg_data(gi))
    check_aa_aa_ordering(graphs)

    print()
    print("=" * 70)
    print("STEP 4 - stress test: many random graph sizes/orders, assert no mismatch")
    print("=" * 70)
    rng = np.random.default_rng(42)
    all_ok = True
    for trial in range(30):
        n_graphs = rng.integers(1, 6)
        sizes = [(int(rng.integers(6, 50)), int(rng.integers(2, 12))) for _ in range(n_graphs)]
        graphs = [to_pyg_data(build_synthetic_graph(ta, taa, seed=trial * 100 + i))
                  for i, (ta, taa) in enumerate(sizes)]
        ok = check_aa_aa_ordering(graphs, verbose=False)
        all_ok = all_ok and ok
    print(f"\nAll 30 randomized trials passed: {all_ok}")

    print()
    print("=" * 70)
    print("STEP 5 - what max_aa_aa_edges actually caps, and why it matters")
    print("=" * 70)
    out_dim, n_ions = 174, 6
    max_aa_aa_edges = out_dim // n_ions
    print(f"out_dim={out_dim}, n_ions={n_ions} -> max_aa_aa_edges={max_aa_aa_edges}")
    print(f"This means peptides with MORE than {max_aa_aa_edges} residues "
          f"(i.e. > {max_aa_aa_edges} peptide bonds) will have their tail bonds "
          f"DROPPED by the `valid = local_rank < self.max_aa_aa_edges` filter -- "
          f"those slots in y are simply never written to (left as 0), even if the "
          f"true target y for that peptide has non-zero values there. Check your "
          f"max peptide length in the dataset against this cap.")

    print()
    print("=" * 70)
    print("STEP 6 - same ordering check on REAL data (optional)")
    print("=" * 70)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_train', type=str,
        default='/home/rrekikdi/Bureau/postdoc/GraphSpectra/hierarchical_dataset/hierarchical_dataset/processed_graphs_train_hcd_hierarchical_dummy')
    parser.add_argument(
        '--root_val', type=str,
        default='/home/rrekikdi/Bureau/postdoc/GraphSpectra/hierarchical_dataset/hierarchical_dataset/processed_graphs_val_hcd_hierarchical_dummy')
    parser.add_argument(
        '--root_test', type=str,
        default='/home/rrekikdi/Bureau/postdoc/GraphSpectra/hierarchical_dataset/hierarchical_dataset/processed_graphs_holdout_hcd_hierarchical_dummy')
    parser.add_argument(
        '--split', type=str, default='train', choices=['train', 'val', 'test'],
        help='which root_* to actually run the check on')
    parser.add_argument('--num_batches', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--skip', action='store_true',
                         help='skip step 6 entirely (only run the synthetic checks above)')
    args = parser.parse_args()

    if args.skip:
        print("Skipped (--skip passed). Synthetic checks above already validated the logic.")
    else:
        root_by_split = {'train': args.root_train, 'val': args.root_val, 'test': args.root_test}
        dataset_root = root_by_split[args.split]
        print(f"Using split='{args.split}' -> {dataset_root}")
        check_real_dataset(
            dataset_root,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
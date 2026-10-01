"""
Per-node-type standardization of hierarchical node features.

Raw AA node features are unscaled (mol_weight ~75-260, num_atom, pKa/pI
~2-12) while atom features are mostly 0/1. After `input_proj`, the
LayerNorm in every `_EdgeTypeGATBlock` then maps all AA nodes onto nearly
the same direction (residue identity is carried only by magnitude), so
AA-pooled readouts become input-independent unless the global node
re-injects signal -- which the "atom_aa" / "aa_only" ablations remove.

Statistics are computed separately for each node type (atom / aa / global),
since the padded layout gives every type its own feature block and shared
columns (charge / energy) have type-specific distributions. Columns that are
constant within a type (e.g. padding zeros) keep std=1, so they map to 0.
The node-type one-hot (last NODE_TYPE_DIM columns) is left untouched: the
structure transform and the model read node types from it.
"""

import torch
from torch_geometric.data import Data

NODE_TYPE_DIM = 3  # last 3 cols of x: [is_atom, is_aa, is_global]


def compute_node_feature_stats(dataset, num_samples=20000):
    """
    Per-node-type mean/std of x[:, :-NODE_TYPE_DIM] over the first
    `num_samples` graphs of `dataset` (read via `dataset.get`, i.e. before any
    transform). Reading the first graphs keeps chunk loading sequential and
    makes the stats deterministic for a given training set.
    Returns (mean, std), each of shape [NODE_TYPE_DIM, num_features].
    """
    n = min(num_samples, dataset.len())
    feat_dim = dataset.get(0).x.shape[1] - NODE_TYPE_DIM
    count = torch.zeros(NODE_TYPE_DIM, 1, dtype=torch.float64)
    total = torch.zeros(NODE_TYPE_DIM, feat_dim, dtype=torch.float64)
    total_sq = torch.zeros(NODE_TYPE_DIM, feat_dim, dtype=torch.float64)

    for i in range(n):
        x = dataset.get(i).x.double()
        node_type = x[:, -NODE_TYPE_DIM:].argmax(dim=1)
        feats = x[:, :-NODE_TYPE_DIM]
        count.index_add_(0, node_type, torch.ones(x.shape[0], 1, dtype=torch.float64))
        total.index_add_(0, node_type, feats)
        total_sq.index_add_(0, node_type, feats * feats)

    count = count.clamp_min(1)
    mean = total / count
    std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
    std[std < 1e-6] = 1.0
    return mean.float(), std.float()


class StandardizeNodeFeatures:
    """
    PyG-style transform applying per-node-type standardization.
    Returns a new Data object: the streaming dataset hands out objects from
    its cached chunk, so mutating `data` in place would re-standardize the
    same graph every time it is read.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data: Data) -> Data:
        node_type_ohe = data.x[:, -NODE_TYPE_DIM:]
        node_type = node_type_ohe.argmax(dim=1)
        feats = (data.x[:, :-NODE_TYPE_DIM] - self.mean[node_type]) / self.std[node_type]
        out = Data(**{key: value for key, value in data})
        out.x = torch.cat([feats, node_type_ohe], dim=1)
        return out

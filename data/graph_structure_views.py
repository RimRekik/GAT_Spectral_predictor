"""
Graph-structure ablation views.

The hierarchical graph built by `hierarchical_streaming_dataset.py` already
contains everything needed for every condition of the graph-structure
ablation ("complete", "atom_aa", "aa_only", "atomic_only"): atom nodes, AA
nodes, a global node, and 4 typed edge sets (atom-atom, atom-aa, aa-aa,
aa-global), all sharing one padded feature layout (see `_pad_features` in
that module).

Rather than re-running RDKit featurization for each condition, this module
prunes an already-built hierarchical `Data` object down to the requested
node/edge subset. Node and edge feature layout is left untouched, so the
*same* hierarchical model class (`model/model_2.py`) can be trained on any
of the four structures unchanged -- only which nodes/edges it can see
changes. That keeps model architecture fixed and isolates the graph
structure as the sole ablated variable, and lets every condition (including
"atomic_only") reuse the same precomputed hierarchical dataset instead of
needing a separate flat `baseline_dataset`.
"""

import torch
from torch_geometric.data import Data

NODE_TYPE_DIM = 3  # last 3 cols of x: [is_atom, is_aa, is_global]
EDGE_TYPE_DIM = 4  # last 4 cols of edge_attr: [atom-atom, atom-aa, aa-aa, aa-global]

ATOM_ATOM, ATOM_AA, AA_AA, AA_GLOBAL = range(4)

STRUCTURES = ("complete", "atom_aa", "aa_only", "atomic_only")

# which edge types survive, per structure (complete keeps all, handled separately)
_KEEP_EDGE_TYPES = {
    "atom_aa": {ATOM_ATOM, ATOM_AA, AA_AA},
    "aa_only": {AA_AA},
    "atomic_only": {ATOM_ATOM},
}


def _node_type_masks(x: torch.Tensor):
    nt = x[:, -NODE_TYPE_DIM:]
    return nt[:, 0].bool(), nt[:, 1].bool(), nt[:, 2].bool()


def restrict_to_structure(data: Data, structure: str) -> Data:
    """
    Returns a new Data object restricted to the requested graph-structure
    condition. `data` must be a hierarchical graph (atom+aa+global nodes,
    4 typed edge sets) as produced by `process_batch_hierarchical`.

    complete     : unchanged (atom + aa + global nodes, all 4 edge types)
    atom_aa      : atom + aa nodes only; atom-atom / atom-aa / aa-aa edges
                   (drops the global node and aa-global edges)
    aa_only      : aa nodes only; aa-aa edges only
                   (drops atom + global nodes and every atom-* edge)
    atomic_only  : atom nodes only; atom-atom edges only
                   (drops aa + global nodes and every aa-*/global edge)
    """
    if structure == "complete":
        return data
    if structure not in STRUCTURES:
        raise ValueError(f"Unknown graph structure '{structure}'. Expected one of {STRUCTURES}.")

    is_atom, is_aa, _is_global = _node_type_masks(data.x)
    if structure == "aa_only":
        keep_node = is_aa
    elif structure == "atomic_only":
        keep_node = is_atom
    else:
        keep_node = is_atom | is_aa

    keep_node_idx = keep_node.nonzero(as_tuple=True)[0]
    remap = torch.full((data.x.shape[0],), -1, dtype=torch.long)
    remap[keep_node_idx] = torch.arange(keep_node_idx.numel())

    edge_type = data.edge_attr[:, -EDGE_TYPE_DIM:].argmax(dim=1)
    keep_edge_types = _KEEP_EDGE_TYPES[structure]
    type_kept = torch.zeros(edge_type.shape[0], dtype=torch.bool)
    for t in keep_edge_types:
        type_kept |= edge_type == t

    src, dst = data.edge_index
    keep_edge = type_kept & keep_node[src] & keep_node[dst]

    out = Data(
        x=data.x[keep_node_idx],
        edge_index=torch.stack([remap[src[keep_edge]], remap[dst[keep_edge]]]),
        edge_attr=data.edge_attr[keep_edge],
        y=data.y,
    )
    if getattr(data, "pos", None) is not None:
        out.pos = data.pos[keep_node_idx]
    return out


def make_structure_transform(structure: str):
    """
    PyG-style transform factory for `Dataset(..., transform=...)`.
    Returns None for "complete" (no-op, avoids a wasted function call per sample).
    """
    if structure == "complete":
        return None
    return lambda data: restrict_to_structure(data, structure)

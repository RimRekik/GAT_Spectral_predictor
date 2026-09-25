from torch_geometric.nn import AttentiveFP, global_mean_pool, GAT, GATv2Conv
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch
import torch.nn as nn

NODE_TYPE_DIM = 3  # [atom, aa, global]
EDGE_TYPE_DIM = 4  # [atom-atom, atom-aa, aa-aa, aa-global]


# ─────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────

def split_edges_by_type(edge_index: torch.Tensor, edge_attr: torch.Tensor, type_dim: int):
    """
    Splits edge_index and edge_attr into type_dim subsets based on the one-hot
    edge type stored in the last type_dim columns of edge_attr.
    Returns a list of (edge_index, edge_attr) tuples, one per edge type.
    """
    type_id = edge_attr[:, -type_dim:].argmax(dim=1)
    return [(edge_index[:, type_id == t], edge_attr[type_id == t]) for t in range(type_dim)]


def get_node_masks(x_raw: torch.Tensor):
    """
    Extracts boolean masks for each node type from the last NODE_TYPE_DIM
    columns of x_raw. Computed once per forward pass and reused everywhere.
    Returns (is_atom, is_aa, is_global).
    """
    nt = x_raw[:, -NODE_TYPE_DIM:]
    return nt[:, 0].bool(), nt[:, 1].bool(), nt[:, 2].bool()


def global_emb_per_edge(h, batch, is_global, src, batch_size):
    """
    Broadcasts the global node embedding to each aa-aa edge.
    Each edge gets the global embedding of the graph it belongs to.
    """
    g = h.new_zeros((batch_size, h.shape[1]))
    g[batch[is_global]] = h[is_global]
    return g[batch[src]]


def local_rank_vectorized(graph_id: torch.Tensor, batch_size: int, device):
    """
    Computes the local index of each aa-aa edge within its own graph,
    without any Python loop over the batch. Used to place ion predictions
    at the correct position in the 174-dim output vector.
    """
    counts  = torch.bincount(graph_id, minlength=batch_size)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                         counts.cumsum(0)[:-1]])
    return torch.arange(graph_id.shape[0], device=device) - offsets[graph_id]


def scatter_ions(out, ion_preds, graph_id, local_rank, max_edges, n_ions):
    """
    Scatters predicted ion intensities into the flat 174-dim output vector.
    Each peptide bond contributes n_ions consecutive values starting at
    local_rank * n_ions. Bonds beyond max_edges are ignored (truncated peptides).
    """
    valid       = local_rank < max_edges
    flat_offset = local_rank[valid] * n_ions
    for k in range(n_ions):
        out[graph_id[valid], flat_offset + k] = ion_preds[valid, k]


# ─────────────────────────────────────────────
# Shared readout (edge-based)
# ─────────────────────────────────────────────

def edge_readout(h, batch, is_global, ei_aa_aa, edge_head,
                 batch_size, hidden_dim, out_dim, n_ions, max_aa_aa_edges, device):
    """
    For each aa-aa edge (peptide bond), concatenates [h_src, h_dst, h_global]
    and predicts n_ions intensities, then scatters them into a [batch_size, out_dim] tensor.

    to_undirected() duplicates each aa-aa edge — we keep only src < dst
    to get exactly one prediction per peptide bond in sequence order.
    """
    out = h.new_zeros((batch_size, out_dim))
    if ei_aa_aa.numel() == 0:
        return out

    keep = ei_aa_aa[0] < ei_aa_aa[1]
    src, dst = ei_aa_aa[0, keep], ei_aa_aa[1, keep]
    if src.numel() == 0:
        return out

    g_emb     = global_emb_per_edge(h, batch, is_global, src, batch_size)
    edge_emb  = torch.cat([h[src], h[dst], g_emb], dim=1)
    ion_preds = edge_head(edge_emb)

    graph_id = batch[src]
    lr       = local_rank_vectorized(graph_id, batch_size, device)
    scatter_ions(out, ion_preds, graph_id, lr, max_aa_aa_edges, n_ions)
    return out


# ─────────────────────────────────────────────
# Building block
# ─────────────────────────────────────────────

class _EdgeTypeGATBlock(nn.Module):
    """
    Single GATv2 convolution dedicated to one edge type, followed by a
    residual connection and layer normalization.
    add_self_loops=False because the graph already encodes self-relations
    through the hierarchical edge structure.
    """
    def __init__(self, hidden_dim, edge_feat_dim, heads=4, dropout=0.2):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            edge_dim=edge_feat_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return x
        # residual: preserve the node's own representation alongside what it learned
        return self.norm(x + self.conv(x, edge_index, edge_attr))


# ─────────────────────────────────────────────
# Shared global readout head
# ─────────────────────────────────────────────

class _GlobalReadout(nn.Module):
    """
    Aggregates AA node embeddings across the whole peptide using a
    SetTransformerAggregation (attention-based pooling), then maps the
    graph-level vector to the 174-dim spectrum output.

    Only AA nodes are used: the MS/MS spectrum depends on peptide bond
    fragmentation, which operates at the residue level. Atom embeddings
    have already been propagated into AA nodes via atom_aa message passing.

    Fallback for the "atomic_only" graph-structure ablation: when no AA
    nodes survive the pruning, pools over atom embeddings instead.
    See `readout_pool_mask` below.
    """
    def __init__(self, hidden_dim, out_dim, dropout):
        super().__init__()
        self.readout = SetTransformerAggregation(channels=hidden_dim, heads=8)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim), nn.LeakyReLU(),  # as for baseline Prosit's regressor output
        )

    def forward(self, h_pool, batch_pool):
        # h_pool : [num_pooled_nodes_in_batch, hidden_dim] (AA nodes, or atom
        #          nodes as a fallback when no AA nodes are present)
        # output : [batch_size, out_dim]
        return self.head(self.readout(h_pool, index=batch_pool))


def readout_pool_mask(is_atom, is_aa):
    """Pool over AA nodes normally; fall back to atom nodes if none remain
    (the "atomic_only" ablation)."""
    return is_aa if is_aa.any() else is_atom


# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────

class Hierachical_Sequential_GAT(nn.Module):
    """
    Hierarchical Sequential GNN with edge-based readout.

    Encoder — bottom-up sequential message passing:
        atom-atom  : refines atom embeddings within each residue (num_layers blocks)
        atom → AA  : aggregates atomic chemistry into the parent residue (1 block)
        AA-AA      : propagates context along the peptide backbone (num_layers blocks)
        AA → global: summarizes all residues into the global context node (1 block)

    Readout — per peptide bond (aa-aa edge):
        For each bond: concat [h_src, h_dst, h_global] → MLP → 6 ion intensities
        Scattered into a sparse 174-dim vector matching the target spectrum layout.

    Limitation: each bond is predicted independently — no cross-bond dependencies.
    See Hierachical_Sequential_GAT_Global for a readout that captures them.
    """
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128,
                 out_dim=174, num_layers=3, heads=4, dropout=0.2,
                 max_aa_aa_edges=None, **kwargs):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.out_dim         = out_dim
        self.n_ions          = 6
        self.max_aa_aa_edges = max_aa_aa_edges or (out_dim // self.n_ions)

        self.input_proj       = nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU())
        self.atom_atom_blocks = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.atom_aa_block    = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
        self.aa_aa_blocks     = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.aa_global_block  = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
        self.edge_head        = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.n_ions), nn.LeakyReLU(),  # as for baseline Prosit's regressor output
        )

    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        is_atom, is_aa, is_global = get_node_masks(x_raw)
        (ei_aa, ea_aa), (ei_atom_aa, ea_atom_aa), \
        (ei_aa_aa, ea_aa_aa), (ei_aa_glob, ea_aa_glob) = split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)

        h = self.input_proj(x_raw)
        for block in self.atom_atom_blocks:
            h = block(h, ei_aa, ea_aa)
        h = self.atom_aa_block(h, ei_atom_aa, ea_atom_aa)
        for block in self.aa_aa_blocks:
            h = block(h, ei_aa_aa, ea_aa_aa)
        h = self.aa_global_block(h, ei_aa_glob, ea_aa_glob)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        return edge_readout(h, batch, is_global, ei_aa_aa, self.edge_head,
                            batch_size, self.hidden_dim, self.out_dim,
                            self.n_ions, self.max_aa_aa_edges, h.device)


class Hierachical_Sequential_GAT_Global(nn.Module):
    """
    Hierarchical Sequential GNN with global readout (strongest sequential model).

    Encoder — identical bottom-up sequential message passing as Hierachical_Sequential_GAT.

    Readout — global (SetTransformer over AA nodes):
        All 174 spectrum bins are predicted simultaneously from a single graph-level
        vector, allowing the model to capture cross-position dependencies
        (e.g. b-ion at position 3 correlates with y-ion at position n-3).
    """
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128,
                 out_dim=174, num_layers=3, heads=4, dropout=0.2,
                 max_aa_aa_edges=None, jumping_knowledge=False, **kwargs):
        super().__init__()
        self.input_proj       = nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU())
        self.atom_atom_blocks = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.atom_aa_block    = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
        self.aa_aa_blocks     = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.aa_global_block  = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
        self.readout_head     = _GlobalReadout(hidden_dim, out_dim, dropout)

        # Jumping Knowledge (opt-in, off by default): concatenate every
        # intermediate layer's node states (instead of using only the last
        # block's output) before the readout, then project back down to
        # hidden_dim. Mitigates over-smoothing -- with many message-passing
        # layers over few nodes (e.g. 19 AA nodes), node embeddings can
        # converge to near-identical vectors by the final layer, erasing
        # the per-peptide signal the readout needs.
        self.jumping_knowledge = jumping_knowledge
        if jumping_knowledge:
            n_layer_outputs = 2 * num_layers + 2  # atom_atom_blocks + atom_aa_block + aa_aa_blocks + aa_global_block
            self.jk_proj = nn.Sequential(
                nn.Linear(n_layer_outputs * hidden_dim, hidden_dim), nn.ReLU(),
            )

    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        is_atom, is_aa, _ = get_node_masks(x_raw)
        (ei_aa, ea_aa), (ei_atom_aa, ea_atom_aa), \
        (ei_aa_aa, ea_aa_aa), (ei_aa_glob, ea_aa_glob) = split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)

        h = self.input_proj(x_raw)
        layer_outputs = []
        for block in self.atom_atom_blocks:
            h = block(h, ei_aa, ea_aa)
            layer_outputs.append(h)
        h = self.atom_aa_block(h, ei_atom_aa, ea_atom_aa)
        layer_outputs.append(h)
        for block in self.aa_aa_blocks:
            h = block(h, ei_aa_aa, ea_aa_aa)
            layer_outputs.append(h)
        h = self.aa_global_block(h, ei_aa_glob, ea_aa_glob)
        layer_outputs.append(h)

        if self.jumping_knowledge:
            h = self.jk_proj(torch.cat(layer_outputs, dim=-1))

        pool = readout_pool_mask(is_atom, is_aa)
        return self.readout_head(h[pool], batch[pool])


class Hierarchical_Cyclic_Sequential_GAT(nn.Module):
    """
    Hierarchical Cyclic GNN with edge-based readout.

    Encoder — cyclic message passing (num_layers full cycles):
        Each cycle runs ONE GAT block per edge type in strict bottom-up order:
            atom-atom → atom-AA → AA-AA → AA-global
        h = LayerNorm(h + delta) after each cycle: the residual connection
        ensures information accumulates across cycles instead of being overwritten.
        delta is pre-allocated outside the loop to avoid repeated GPU allocations.

    Readout — same edge-based readout as Hierachical_Sequential_GAT.
    """
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128,
                 out_dim=174, num_layers=3, heads=4, dropout=0.2,
                 max_aa_aa_edges=None, **kwargs):
        super().__init__()
        self.hidden_dim      = hidden_dim
        self.out_dim         = out_dim
        self.num_layers      = num_layers
        self.n_ions          = 6
        self.max_aa_aa_edges = max_aa_aa_edges or (out_dim // self.n_ions)

        self.input_proj    = nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU())
        self.gat_atom_atom = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.gat_atom_aa   = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.gat_aa_aa     = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.gat_aa_global = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.cycle_norms   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.edge_head     = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, self.n_ions), nn.LeakyReLU(),  # as for baseline Prosit's regressor output
        )

    def _cycle(self, h, cycle, edges, is_atom, is_aa, is_global, delta):
        """
        One full hierarchical cycle: runs 4 GAT blocks sequentially,
        accumulates their outputs into delta, then applies residual + norm.
        delta is zeroed in-place to avoid reallocation at each cycle.
        """
        (ei_aa, ea_aa), (ei_atom_aa, ea_atom_aa), \
        (ei_aa_aa, ea_aa_aa), (ei_aa_glob, ea_aa_glob) = edges

        out_aa    = self.gat_atom_atom[cycle](h, ei_aa,      ea_aa)
        out_atoaa = self.gat_atom_aa[cycle]  (h, ei_atom_aa, ea_atom_aa)
        out_aaaa  = self.gat_aa_aa[cycle]    (h, ei_aa_aa,   ea_aa_aa)
        out_glob  = self.gat_aa_global[cycle](h, ei_aa_glob, ea_aa_glob)

        # each node type receives messages only from its relevant edge type(s)
        delta.zero_()
        delta[is_atom]   = out_aa[is_atom]
        delta[is_aa]     = out_atoaa[is_aa] + out_aaaa[is_aa]
        delta[is_global] = out_glob[is_global]
        return self.cycle_norms[cycle](h + delta)

    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        is_atom, is_aa, is_global = get_node_masks(x_raw)
        edges = split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)
        _, _, (ei_aa_aa, _), _ = edges

        h     = self.input_proj(x_raw)
        delta = torch.empty_like(h)  # allocated once, zeroed in-place each cycle
        for cycle in range(self.num_layers):
            h = self._cycle(h, cycle, edges, is_atom, is_aa, is_global, delta)

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        return edge_readout(h, batch, is_global, ei_aa_aa, self.edge_head,
                            batch_size, self.hidden_dim, self.out_dim,
                            self.n_ions, self.max_aa_aa_edges, h.device)


class Hierarchical_Cyclic_Sequential_GAT_Global(nn.Module):
    """
    Hierarchical Cyclic GNN with global readout.

    Encoder — same cyclic bottom-up message passing as Hierarchical_Cyclic_Sequential_GAT.

    Readout — global (SetTransformer over AA nodes):
        Predicts all 174 bins simultaneously, capturing cross-position dependencies.
        - cyclic encoder: information refines iteratively across all hierarchy levels
        - global readout: all spectrum positions influence each other at prediction time

    Model-ablation results (lr=5e-4, hidden_dim=256, num_layers=7, wandb project
    graph-spectra, held-out test_loss) put this within noise of, or slightly
    behind, the sequential encoder + global readout (Hierachical_Sequential_GAT_Global
    / "hierarchical_GAT"): 0.1287±0.0022 vs 0.1235±0.0011 at 150k steps,
    0.1092±0.0026 vs 0.1049±0.0017 at 600k -- despite this encoder running ~1.75x
    more GAT blocks per layer. The global-vs-local readout choice is the effect
    that actually matters (~0.012-0.019); cyclic vs sequential is not the one to
    lead with.
    """
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128,
                 out_dim=174, num_layers=3, heads=4, dropout=0.2,
                 max_aa_aa_edges=None, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.input_proj    = nn.Sequential(nn.Linear(node_feat_dim, hidden_dim), nn.ReLU())
        self.gat_atom_atom = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.gat_atom_aa   = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.gat_aa_aa     = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.gat_aa_global = nn.ModuleList([_EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout) for _ in range(num_layers)])
        self.cycle_norms   = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.readout_head  = _GlobalReadout(hidden_dim, out_dim, dropout)

    def _cycle(self, h, cycle, edges, is_atom, is_aa, is_global, delta):
        """One full hierarchical cycle — see Hierarchical_Cyclic_Sequential_GAT._cycle."""
        (ei_aa, ea_aa), (ei_atom_aa, ea_atom_aa), \
        (ei_aa_aa, ea_aa_aa), (ei_aa_glob, ea_aa_glob) = edges

        out_aa    = self.gat_atom_atom[cycle](h, ei_aa,      ea_aa)
        out_atoaa = self.gat_atom_aa[cycle]  (h, ei_atom_aa, ea_atom_aa)
        out_aaaa  = self.gat_aa_aa[cycle]    (h, ei_aa_aa,   ea_aa_aa)
        out_glob  = self.gat_aa_global[cycle](h, ei_aa_glob, ea_aa_glob)

        delta.zero_()
        delta[is_atom]   = out_aa[is_atom]
        delta[is_aa]     = out_atoaa[is_aa] + out_aaaa[is_aa]
        delta[is_global] = out_glob[is_global]
        return self.cycle_norms[cycle](h + delta)

    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        is_atom, is_aa, is_global = get_node_masks(x_raw)
        edges = split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)

        h     = self.input_proj(x_raw)
        delta = torch.empty_like(h)  # allocated once, zeroed in-place each cycle
        for cycle in range(self.num_layers):
            h = self._cycle(h, cycle, edges, is_atom, is_aa, is_global, delta)

        return self.readout_head(h[readout_pool_mask(is_atom, is_aa)], batch[readout_pool_mask(is_atom, is_aa)])


# ─────────────────────────────────────────────
# Registry — model selection from config
# ─────────────────────────────────────────────

MODEL_REGISTRY = {
    "seq_gat":           Hierachical_Sequential_GAT,
    "seq_gat_global":    Hierachical_Sequential_GAT_Global,
    "cyclic_gat":        Hierarchical_Cyclic_Sequential_GAT,
    "cyclic_gat_global": Hierarchical_Cyclic_Sequential_GAT_Global,
}


def build_model(config: dict) -> nn.Module:
    """
    Builds a model from a config dict (typically vars(args) from argparse).

    Usage in main.py:
        model = build_model(vars(args))
    """
    name = config.get("model")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**config)
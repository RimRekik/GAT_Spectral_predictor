from torch_geometric.nn import AttentiveFP, global_mean_pool, GAT, GATv2Conv
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch
import torch.nn as nn

NODE_TYPE_DIM = 3   # one-hot, derniers chiffres de x  : [atom, aa, global]
EDGE_TYPE_DIM = 4   # one-hot, derniers chiffres de edge_attr : [atom_atom, atom_aa, aa_aa, aa_global]

class AttentiveFPGraphRegressor(nn.Module):
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128, out_dim=174,num_layers=3, num_timesteps=2):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_feat_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=0.2
        )
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        # Get node embeddings
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)

        # Map to graph-level output
        out = self.lin(x)# [batch_size x out_dim]
        return out

class BaselineGAT(nn.Module):
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128, out_dim=174,num_layers=3):
        super().__init__()

        self.gnn = GAT(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            v2=True,
            edge_dim=edge_feat_dim,
        )

        self.readout = SetTransformerAggregation(channels=hidden_dim, heads=8)

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        x_read = self.readout(x,index=data.batch)
        out = self.lin(x_read)
        return out

class _EdgeTypeGATBlock(nn.Module):
    """
    Un bloc GATv2 appliqué uniquement sur un sous-ensemble d'arêtes
    (filtrées par leur one-hot edge_type), avec connexion résiduelle.
    Permet de donner un traitement (poids) distinct à chaque "étage"
    de la hiérarchie (atom-atom, atom-aa, aa-aa, aa-global).
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
        out = self.conv(x, edge_index, edge_attr)
        return self.norm(x + out)

'''
class Hierachical_GAT(nn.Module):
    """
    GNN hiérarchique en 2 temps :

    1) Message-passing ascendant, étage par étage, en routant chaque
       message uniquement sur le sous-graphe correspondant (filtré via
       le one-hot du type d'arête déjà présent dans edge_attr) :
           atom-atom  -> raffine les embeddings atomes entre eux
           atom-aa    -> remonte l'info atomes vers leur résidu (bipartite)
           aa-aa      -> propage le long du squelette peptidique
           aa-global  -> remonte vers le noeud global (contexte charge/energie)

    2) Lecture (readout) localisée : pour chaque arête aa-aa (= chaque
       liaison peptidique), on combine les embeddings des 2 noeuds AA
       + le noeud global (contexte) via un MLP -> 6 valeurs d'intensité
       (les 6 types d'ions de fragmentation). On assemble ensuite tous
       ces vecteurs 6-dim dans le bon emplacement d'un vecteur 174-dim,
       le reste étant mis à 0 (comme dans la cible y).
    """

    def __init__(
        self,
        node_feat_dim=3,
        edge_feat_dim=3,
        hidden_dim=128,
        out_dim=174,
        num_layers=3,
        heads=4,
        dropout=0.2,
        max_aa_aa_edges=None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_ions = 6  # nb de types d'ions de fragmentation par liaison aa-aa
        # nb max de liaisons aa-aa représentables dans la sortie (174 // 6 par défaut)
        self.max_aa_aa_edges = max_aa_aa_edges or (out_dim // self.n_ions)

        # encodeur d'entrée : projette les features brutes (hétérogènes
        # selon le type de noeud, déjà paddées dans le dataset) dans
        # un espace caché commun
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )

        # un bloc GAT indépendant par étage hiérarchique, répété num_layers fois
        self.atom_atom_blocks = nn.ModuleList([
            _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.atom_aa_block = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
        self.aa_aa_blocks = nn.ModuleList([
            _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.aa_global_block = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)

        # tete de prediction par arete aa-aa :
        # [emb_aa_i | emb_aa_j | emb_global_du_graphe] -> 6 intensites
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.n_ions),
        )

    @staticmethod
    def _split_edges_by_type(edge_index, edge_attr, type_dim):
        """
        Sépare edge_index/edge_attr en sous-ensembles selon le one-hot
        de type d'arête stocké dans les `type_dim` dernières colonnes
        de edge_attr. Retourne un dict {type_idx: (edge_index, edge_attr)}.
        """
        type_onehot = edge_attr[:, -type_dim:]
        type_id = type_onehot.argmax(dim=1)

        out = {}
        for t in range(type_dim):
            mask = type_id == t
            out[t] = (edge_index[:, mask], edge_attr[mask])
        return out

    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_type_onehot = x_raw[:, -NODE_TYPE_DIM:]  # [atom, aa, global]
        is_atom = node_type_onehot[:, 0].bool()
        is_aa = node_type_onehot[:, 1].bool()
        is_global = node_type_onehot[:, 2].bool()

        # --- split des arêtes par type (0=atom_atom, 1=atom_aa, 2=aa_aa, 3=aa_global) ---
        edges_by_type = self._split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)
        ei_atom_atom, ea_atom_atom = edges_by_type[0]
        ei_atom_aa, ea_atom_aa = edges_by_type[1]
        ei_aa_aa, ea_aa_aa = edges_by_type[2]
        ei_aa_global, ea_aa_global = edges_by_type[3]

        # --- encodage initial commun ---
        h = self.input_proj(x_raw)

        # --- étage 1 : atome <-> atome (plusieurs couches) ---
        for block in self.atom_atom_blocks:
            h = block(h, ei_atom_atom, ea_atom_atom)

        # --- étage 2 : atome -> AA (remonte l'info atomique dans chaque résidu) ---
        h = self.atom_aa_block(h, ei_atom_aa, ea_atom_aa)

        # --- étage 3 : AA <-> AA (propagation le long du squelette peptidique) ---
        for block in self.aa_aa_blocks:
            h = block(h, ei_aa_aa, ea_aa_aa)

        # --- étage 4 : AA -> global (contexte charge / énergie de collision) ---
        h = self.aa_global_block(h, ei_aa_global, ea_aa_global)

        # --- lecture par arête aa-aa ---
        # to_undirected() (appliqué dans le dataset) duplique chaque arête
        # aa-aa en (i->j) et (j->i) : on ne garde qu'un seul sens (src < dst)
        # pour avoir une seule prédiction par liaison peptidique, dans un
        # ordre stable correspondant à la position dans la séquence.
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = h.new_zeros((batch_size, self.out_dim))

        if ei_aa_aa.numel() > 0:
            keep = ei_aa_aa[0] < ei_aa_aa[1]
            src, dst = ei_aa_aa[0, keep], ei_aa_aa[1, keep]
        else:
            src = dst = ei_aa_aa.new_zeros((0,), dtype=torch.long)

        if src.numel() > 0:
            # contexte global du graphe correspondant à chaque noeud (pour broadcast par arête)
            global_emb_per_graph = h.new_zeros((batch_size, self.hidden_dim))
            global_emb_per_graph[batch[is_global]] = h[is_global]
            global_emb_per_edge = global_emb_per_graph[batch[src]]

            edge_emb = torch.cat([h[src], h[dst], global_emb_per_edge], dim=1)
            ion_preds = self.edge_head(edge_emb)  # [num_aa_aa_edges_total_batch, 6]

            # position de chaque arête aa-aa dans la séquence du peptide
            # (rang local au sein de son propre graphe, src < dst donc
            # l'ordre suit la construction du dataset : idx, idx+1, ...),
            # pour la placer au bon emplacement du vecteur 174-dim
            graph_id = batch[src]
            local_rank = torch.zeros_like(graph_id)
            for g in range(batch_size):
                g_mask = graph_id == g
                n_edges_g = int(g_mask.sum().item())
                local_rank[g_mask] = torch.arange(n_edges_g, device=h.device)

            valid = local_rank < self.max_aa_aa_edges
            flat_offset = local_rank[valid] * self.n_ions
            for k in range(self.n_ions):
                out[graph_id[valid], flat_offset + k] = ion_preds[valid, k]

        return out
'''

# ============================================================
# NOUVEAU : Hierarchical_GAT avec SetTransformer readout
# ============================================================
class Hierachical_GAT(nn.Module):
    """
    Encodeur hiérarchique (inchangé) :
        atom-atom  → raffine les embeddings atomes
        atom-aa    → remonte l'info atomique vers les résidus
        aa-aa      → propage le long du squelette peptidique
        aa-global  → agrège vers le nœud de contexte (charge/énergie)
 
    Readout (NOUVEAU) :
        SetTransformerAggregation sur les nœuds AA uniquement
        → vecteur de graphe → Linear(hidden_dim, 174)
     
    """
 
    def __init__(
        self,
        node_feat_dim=3,
        edge_feat_dim=3,
        hidden_dim=128,
        out_dim=174,
        num_layers=3,
        heads=4,
        dropout=0.2,
        max_aa_aa_edges=None,   # gardé pour compatibilité, non utilisé
    ):
        super().__init__()
 
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
 
        # projection d'entrée
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
        )
 
        self.atom_atom_blocks = nn.ModuleList([
            _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.atom_aa_block  = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
        self.aa_aa_blocks   = nn.ModuleList([
            _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.aa_global_block = _EdgeTypeGATBlock(hidden_dim, edge_feat_dim, heads, dropout)
 
        self.readout = SetTransformerAggregation(channels=hidden_dim, heads=8)
 
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
 
    @staticmethod
    def _split_edges_by_type(edge_index, edge_attr, type_dim):
        type_onehot = edge_attr[:, -type_dim:]
        type_id = type_onehot.argmax(dim=1)
        out = {}
        for t in range(type_dim):
            mask = type_id == t
            out[t] = (edge_index[:, mask], edge_attr[mask])
        return out
 
    def forward(self, data):
        x_raw, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
 
        node_type_onehot = x_raw[:, -NODE_TYPE_DIM:]
        is_aa = node_type_onehot[:, 1].bool()
 
        edges_by_type = self._split_edges_by_type(edge_index, edge_attr, EDGE_TYPE_DIM)
        ei_atom_atom, ea_atom_atom = edges_by_type[0]
        ei_atom_aa,   ea_atom_aa   = edges_by_type[1]
        ei_aa_aa,     ea_aa_aa     = edges_by_type[2]
        ei_aa_global, ea_aa_global = edges_by_type[3]
 
        h = self.input_proj(x_raw)
 
        for block in self.atom_atom_blocks:
            h = block(h, ei_atom_atom, ea_atom_atom)
 
        h = self.atom_aa_block(h, ei_atom_aa, ea_atom_aa)
 
        for block in self.aa_aa_blocks:
            h = block(h, ei_aa_aa, ea_aa_aa)
 
        h = self.aa_global_block(h, ei_aa_global, ea_aa_global)
 
        h_aa       = h[is_aa]               
        batch_aa   = batch[is_aa]          
 
        graph_emb  = self.readout(h_aa, index=batch_aa)  
        out = self.head(graph_emb)         
        return out
 

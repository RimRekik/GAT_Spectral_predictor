import numpy as np
from typing import Union, List
import pandas as pd
import torch
import ast
import h5py
import os
import bisect
import random

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import Dataset

from multiprocessing import Pool, cpu_count
from functools import partial
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from data.graph_creation_utils import *


# =========================
# GLOBALS for multiprocessing
# =========================
SEQ = None
INTY = None
CHARGE = None
ENERGY = None
GLOBAL_FEATURE_DIM = 64
EDGE_TYPE_DIM = 4
# [atom_atom, atom_aa, aa_aa, aa_global]
NODE_TYPE_DIM = 3
# [atom, aa, global]


def _pad_features(features, before_dim, after_dim):
    """Pad heterogeneous node features into one homogeneous feature space."""
    return np.concatenate([
        np.zeros((features.shape[0], before_dim), dtype=np.float32),
        features.astype(np.float32),
        np.zeros((features.shape[0], after_dim), dtype=np.float32),
    ], axis=1)


def _repeat_global_features(global_features, num_nodes):
    return np.tile(global_features.reshape(1, -1), (num_nodes, 1)).astype(np.float32)


def _edge_array(edges):
    if len(edges) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)

# =========================
# Worker initializer
# =========================
def init_worker(seq, inty, charge, energy):
    global SEQ, INTY, CHARGE, ENERGY
    SEQ = seq
    INTY = inty
    CHARGE = charge
    ENERGY = energy


# =========================
# Worker function
# =========================

def process_one(i, with_position=True):
    try :
        seq = ''.join(int_to_aa_dict[n] for n in SEQ[i].tolist())
        inty = INTY[i]
        charge_ohe = CHARGE[i]
        energy = ENERGY[i]

        charge = np.argmax(charge_ohe)

        # ---- molecule ----
        if '(ox)' in seq:
            mol = seq_to_mol_with_ox(seq)
        else:
            mol = Chem.MolFromSequence(seq)

        if mol is None:
            return None

        # ---- hierarchical node features ----
        global_features = np.concatenate([charge_ohe, energy]).astype(np.float32)
        excluded_feature = ['conformation']
        if with_position:
            excluded_feature=None

        mol_feats = precompute_mol_features(mol,exclude_feature=excluded_feature)

        # AA node features
        x_atom = get_node_features(
            mol,
            mol_feats=mol_feats,
        )
        x_aa = get_aa_node_features(mol)

        # Global node
        x_global = np.zeros((1, 0), dtype=np.float32)

        atom_dim = x_atom.shape[1]
        aa_dim = x_aa.shape[1]

        x_atom = _pad_features(x_atom, 0, aa_dim)
        x_aa = _pad_features(x_aa, atom_dim, 0)
        x_global = _pad_features(x_global, atom_dim + aa_dim, 0)

        #add global feature (collision energy + charge)
        x_atom = np.concatenate([
            x_atom,
            _repeat_global_features(global_features, x_atom.shape[0]),
        ], axis=1)

        x_aa = np.concatenate([
            x_aa,
            _repeat_global_features(global_features, x_aa.shape[0]),
        ], axis=1)

        x_global = np.concatenate([
            x_global,
            _repeat_global_features(global_features, x_global.shape[0]),
        ], axis=1)


        #add node type
        node_type = np.concatenate([
            np.tile([1, 0, 0], (x_atom.shape[0], 1)),
            np.tile([0, 1, 0], (x_aa.shape[0], 1)),
            np.tile([0, 0, 1], (x_global.shape[0], 1)),
        ], axis=0).astype(np.float32)

        #concatenate to a unified feature vector
        x = np.concatenate([
            np.concatenate([x_atom, x_aa, x_global], axis=0),
            node_type,
        ], axis=1)

        #  atom - atom edges (undirected edges are added later)
        edges_atom_atom = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

        # atom - AA edges
        total_atom = mol.GetNumAtoms()
        total_aa = x_aa.shape[0]
        global_idx = total_atom + total_aa

        if with_position:
            pos = np.concatenate([
                get_atom_positions(mol, mol_feats=mol_feats),
                get_aa_node_positions(mol, mol_feats=mol_feats),
                np.zeros((1, 3), dtype=np.float32),
            ], axis=0)

        residue_numbers = sorted({
            atom.GetMonomerInfo().GetResidueNumber() for atom in mol.GetAtoms()
        })
        residue_to_node = {
            residue_number: total_atom + idx
            for idx, residue_number in enumerate(residue_numbers)
        }
        edges_atom_aa = [
            (atom.GetIdx(), residue_to_node[atom.GetMonomerInfo().GetResidueNumber()])
            for atom in mol.GetAtoms()
        ]

        # AA - AA edges
        edges_aa_aa = [
            (total_atom + idx, total_atom + idx + 1)
            for idx in range(total_aa - 1)
        ]

        # AA - global edges

        edges_aa_global = [
            (total_atom + idx, global_idx)
            for idx in range(total_aa)
        ]

        edges = np.concatenate([
            _edge_array(edges_atom_atom),
            _edge_array(edges_atom_aa),
            _edge_array(edges_aa_aa),
            _edge_array(edges_aa_global),
        ], axis=0)
        edge_index = edges.T

        bond_dim = get_edge_dim()
        #physico-chemical prop for atom-atom, zero for others + one hot encoding edge type

        edge_attr_atom_atom = np.concatenate([
            get_edge_features(mol),
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

        edge_attr = np.concatenate([
            edge_attr_atom_atom,
            edge_attr_atom_aa,
            edge_attr_aa_aa,
            edge_attr_aa_global,
        ], axis=0)

        # ---- labels ----
        #only for aa-aa bonds
        y = np.array(inty, dtype=np.float32)

        if with_position :
            return {
                "x": x,
                "pos": pos,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "y": y
            }
        else :
            return {
                "x": x,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "y": y
            }

    except Exception as e:
        print(f"[Worker ERROR] index {i}: {e}")
        return None


# =========================
# Batch processing with multiprocessing
# =========================
def process_batch_hierarchical(start, end, sequence, intensity, charge, energy, with_position=True):
    try:
        seq_batch = sequence[start:end]
        inty_batch = intensity[start:end]
        charge_batch = charge[start:end]
        energy_batch = energy[start:end]

        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(seq_batch, inty_batch, charge_batch, energy_batch)
        ) as pool:
            results = pool.map(partial(process_one, with_position=with_position), range(end - start))

        data_list = []
        for r in results:
            if r is None:
                continue

            try:
                edge_index = torch.from_numpy(r["edge_index"]).long()
                edge_attr = torch.from_numpy(r["edge_attr"]).float()

                edge_index, edge_attr = to_undirected(edge_index, edge_attr)
                if with_position :
                    data = Data(
                        x=torch.from_numpy(r["x"]).float(),
                        pos=torch.from_numpy(r["pos"]).float(),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=torch.from_numpy(r["y"]).float()
                    )
                else :
                    data = Data(
                        x=torch.from_numpy(r["x"]).float(),
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=torch.from_numpy(r["y"]).float()
                    )

                data_list.append(data)

            except Exception as e:
                print(f"[Post-process ERROR]: {e}")
                continue

        return data_list

    except Exception as e:
        print(f"[Batch ERROR] {start}-{end}: {e}")
        return []



class HierarchicalStreamingSpectraDataset(Dataset):
    def __init__(self, root, transform=None):
        # `transform` (e.g. make_structure_transform(...) from
        # graph_structure_views.py) is applied by PyG's base __getitem__
        # after self.get(idx) -- used for the graph-structure ablation.
        super().__init__(root, transform=transform)

        self.root = root
        meta_file = os.path.join(root, "meta.txt")

        self.chunk_files = []
        self.cumulative_sizes = []

        total = 0

        print("Loading metadata...")

        with open(meta_file, "r") as f:
            for line in f:
                path, size = line.strip().split(",")
                size = int(size)

                self.chunk_files.append(os.path.join(root, os.path.basename(path)))
                total += size
                self.cumulative_sizes.append(total)

        self.total_len = total

        self.cache = None
        self.current_chunk_idx = -1

        print(f"Total graphs: {self.total_len}")

    def chunk_shuffle(self): #exclude last chunk since it is incomplete
        rest = self.chunk_files[:-1]
        random.shuffle(rest)
        self.chunk_files[:-1] = rest

    def len(self):
        return self.total_len

    def get(self, idx):
        # fast binary search
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        start = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
        local_idx = idx - start

        # load chunk if needed
        if chunk_idx != self.current_chunk_idx:
            self.cache = torch.load(self.chunk_files[chunk_idx],weights_only=False)
            self.current_chunk_idx = chunk_idx

        data = self.cache[local_idx]

        return data

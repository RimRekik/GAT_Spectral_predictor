import torch
from torch.nn.functional import dropout
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import os
import itertools
import numpy as np
from data.hierarchical_streaming_dataset import HierarchicalStreamingSpectraDataset
from data.graph_structure_views import make_structure_transform
from data.feature_standardization import compute_node_feature_stats, StandardizeNodeFeatures
from torch_geometric.transforms import Compose
from model.model import BaselineGAT, BondBreakPredictor
from model.model_2 import (Hierachical_Sequential_GAT, Hierachical_Sequential_GAT_Global,
                           Hierarchical_Cyclic_Sequential_GAT, Hierarchical_Cyclic_Sequential_GAT_Global)
from model.losses import masked_spectral_distance
from config import load_args
import pandas as pd

def infinite_loader(loader):
    """Creates an infinite dataloader iterator."""
    while True:
        loader.dataset.chunk_shuffle()  # reshuffle chunks every pass
        for batch in loader:
            yield batch


def train_step(data):
    model.train()

    data = data.to(device)

    optimizer.zero_grad()

    out = model(data)

    loss = masked_spectral_distance(
        data.y.view(data.num_graphs, -1),
        out
    )

    loss_mean = loss.mean()
    median = loss.median()
    loss_mean.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_mean.item(), median.item(), data.num_graphs


@torch.inference_mode()
def evaluate(loader, split="val", csv_path=None):

    model.eval()

    sample_losses = []

    save_csv = csv_path is not None

    if save_csv:
        with open(csv_path, "w") as f:
            f.write("Pred,True\n")

    pbar = tqdm(loader, desc=f"[{split.upper()}]", leave=False)

    for data in pbar:
        data = data.to(device)

        out = model(data)

        batch_loss = masked_spectral_distance(
            data.y.view(data.num_graphs, -1),
            out
        )

        mean_loss = batch_loss.mean()
        pbar.set_postfix(loss=mean_loss.item())

        sample_losses.extend(batch_loss.cpu().numpy())

        if save_csv:
            pred = out.cpu().numpy()
            target = data.y.view(data.num_graphs, -1).cpu().numpy()

            with open(csv_path, "a") as f:
                for p, t in zip(pred, target):
                    f.write(
                        f"\"{p.tolist()}\",\"{t.tolist()}\"\n"
                    )

    return {
        "mean": np.mean(sample_losses),
        "median": np.median(sample_losses),
    }


if __name__ == '__main__':

    args = load_args()

    # -----------------------
    # WandB init
    # -----------------------
    os.environ["WANDB_API_KEY"] = 'b4a27ac6b6145e1a5d0ee7f9e2e8c20bd101dccd'
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = os.path.abspath("./wandb_run")

    wandb.init(
        project="graph-spectra",
        config={
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_timesteps": args.num_timesteps,
            "model_type": args.model_type,
            "graph_structure": args.graph_structure,
            "dropout":args.dropout,
            "dataset_train": args.root_train,
            "scheduler":args.scheduler,
            "activation":args.activation,
            "load_weights":args.load_weights,
            "jumping_knowledge":args.jumping_knowledge,
            "standardize_features":args.standardize_features,
            "stats_samples":args.stats_samples,
        }
    )

    config = wandb.config

    # -----------------------
    # Data
    # -----------------------
    # complete / atom_aa / aa_only / atomic_only: all share the same hierarchical
    # dataset, pruned to the requested node/edge subset via a PyG transform
    # (see graph_structure_views.py). root_* must point to a hierarchical
    # dataset (hierarchical_streaming_dataset.py) in every case.
    # Feature standardization runs before pruning so every structure uses the
    # same statistics (computed on the full hierarchical train graphs).
    train_dataset = HierarchicalStreamingSpectraDataset(root=args.root_train)
    transforms = []
    if config.standardize_features:
        feat_mean, feat_std = compute_node_feature_stats(train_dataset, config.stats_samples)
        transforms.append(StandardizeNodeFeatures(feat_mean, feat_std))
        print('Node feature stats computed on', min(config.stats_samples, len(train_dataset)), 'train graphs.')
    structure_transform = make_structure_transform(config.graph_structure)
    if structure_transform is not None:
        transforms.append(structure_transform)
    transform = Compose(transforms) if transforms else None

    train_dataset.transform = transform
    val_dataset = HierarchicalStreamingSpectraDataset(root=args.root_val, transform=transform)
    test_dataset = HierarchicalStreamingSpectraDataset(root=args.root_test, transform=transform)

    print('Data loaded.')
    print(
        'Data dim -- node : ',
        train_dataset[0].x.shape,
        ' edge : ',
        train_dataset[0].edge_attr.shape,
        ' y : ',
        train_dataset[0].y.shape
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=6,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=6,
        pin_memory=True
    )

    train_iter = infinite_loader(train_loader)

    if config.model_type == "GAT":
        model = BaselineGAT(
            node_feat_dim=train_dataset[0].x.shape[1],
            edge_feat_dim=train_dataset[0].edge_attr.shape[1],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            out_dim=174,
            dropout=config.dropout,
            activation=config.activation=='relu'
        )
    elif config.model_type == "local_GAT":
        model = BondBreakPredictor(
            node_feat_dim=train_dataset[0].x.shape[1],
            edge_feat_dim=train_dataset[0].edge_attr.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            out_dim=174)

    elif config.model_type == "hierarchical_GAT":
        model = Hierachical_Sequential_GAT_Global(node_feat_dim=train_dataset[0].x.shape[1],
                    edge_feat_dim=train_dataset[0].edge_attr.shape[1], hidden_dim=args.hidden_dim,
                 out_dim=174, num_layers=args.num_layers, heads=4, dropout=config.dropout,
                 max_aa_aa_edges=None, jumping_knowledge=args.jumping_knowledge,)

    elif config.model_type == "local_hierarchical_GAT":
        model = Hierachical_Sequential_GAT(node_feat_dim=train_dataset[0].x.shape[1],
                    edge_feat_dim=train_dataset[0].edge_attr.shape[1], hidden_dim=args.hidden_dim,
                 out_dim=174, num_layers=args.num_layers, heads=4, dropout=config.dropout,
                 max_aa_aa_edges=None,)

    elif config.model_type == "local_cyclic_GAT":
        model = Hierarchical_Cyclic_Sequential_GAT(node_feat_dim=train_dataset[0].x.shape[1],
                    edge_feat_dim=train_dataset[0].edge_attr.shape[1], hidden_dim=args.hidden_dim,
                 out_dim=174, num_layers=args.num_layers, heads=4, dropout=config.dropout,
                 max_aa_aa_edges=None,)

    elif config.model_type == "cyclic_GAT":
        model = Hierarchical_Cyclic_Sequential_GAT_Global(node_feat_dim=train_dataset[0].x.shape[1],
                    edge_feat_dim=train_dataset[0].edge_attr.shape[1], hidden_dim=args.hidden_dim,
                 out_dim=174, num_layers=args.num_layers, heads=4, dropout=config.dropout,
                 max_aa_aa_edges=None,)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used:', device)

    model = model.to(device)

    print('Model loaded.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,  # validation evaluations, not training steps
            min_lr=1e-6,
            threshold=1e-4,
            threshold_mode="rel"
        )
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.max_steps)

    current_lr = config.lr


    print('Optimizer loaded.')

    # save path dir creation
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # -----------------------
    # Step-based training loop
    # -----------------------
    best_loss = float('inf')

    if config.load_weights != None:
        print('Loading weights from: ', config.load_weights)
        model.load_state_dict(torch.load(config.load_weights, weights_only=True))

    if config.max_steps > 0 :
        print('Starting Training...')

        pbar = tqdm(range(1, config.max_steps + 1), desc="Training")

        for step in pbar:

            data = next(train_iter)

            train_loss, median, batch_size = train_step(data)

            # cosine is annealed over max_steps, so it must step every training step
            if config.scheduler == 'cosine':
                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

            pbar.set_postfix(train_loss=train_loss, median = median)

            # -----------------------
            # Logging
            # -----------------------
            if step%100==0 : #to limit wandb sync size
                wandb.log({
                    "step": step,
                    "train_loss": train_loss
                })

            # -----------------------
            # Validation
            # -----------------------
            if step % config.eval_every == 0:

                val_results = evaluate(val_loader, split="val")

                val_loss = val_results["mean"]
                val_median = val_results["median"]

                if config.scheduler == 'plateau':
                    scheduler.step(val_loss)
                    current_lr = optimizer.param_groups[0]["lr"]

                print(
                    f"\nStep {step:06d}"
                    f" | Train: {train_loss:.4f}"
                    f" | Val Mean: {val_loss:.4f}"
                    f" | Val Median: {val_median:.4f}"
                    f" | LR: {current_lr:.2e}"
                )

                wandb.log({
                    "step": step,
                    "val_mean": val_loss,
                    "val_median": val_median,
                    "lr": current_lr,
                })

                # -----------------------
                # Save best model
                # -----------------------
                if val_loss < best_loss:
                    best_loss = val_loss

                    torch.save(model.state_dict(), args.save_path)

                    print(f"Best model saved at step {step}")

    # -----------------------
    # Test
    # -----------------------


    if config.max_steps > 0 :
        print("Loading best model...")
        model.load_state_dict(torch.load(args.save_path, weights_only=True))

    csv_path = os.path.splitext(args.save_path)[0] + "_predictions.csv"

    test_results = evaluate(
        test_loader,
        split="test",
        csv_path=csv_path,
    )

    print("Test Mean:", test_results["mean"])
    print("Test Median:", test_results["median"])

    wandb.log({
        "test_loss": test_results["mean"],
        "test_median": test_results["median"],
    })

    wandb.finish()
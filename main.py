import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import os
import itertools
import numpy as np
from data.streaming_dataset import StreamingSpectraDataset
from data.hierarchical_streaming_dataset import HierarchicalStreamingSpectraDataset
from model.model import AttentiveFPGraphRegressor, BaselineGAT, Hierachical_GAT
from model.losses import masked_spectral_distance
from config import load_args


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
        out,
        data.y.view(data.num_graphs, -1)
    )

    loss.backward()
    optimizer.step()

    return loss.item(), data.num_graphs


@torch.inference_mode()
def evaluate(loader, split="val"):
    model.eval()
    total_loss = 0

    pbar = tqdm(loader, desc=f"[{split.upper()}]")

    for data in pbar:
        data = data.to(device)

        out = model(data)

        loss = masked_spectral_distance(
            out,
            data.y.view(data.num_graphs, -1)
        )

        total_loss += loss.item() * data.num_graphs

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


if __name__ == '__main__':

    args = load_args()

    # -----------------------
    # WandB init
    # -----------------------
    os.environ["WANDB_API_KEY"] = 'b4a27ac6b6145e1a5d0ee7f9e2e8c20bd101dccd'
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = os.path.abspath("./wandb_run")

    wandb.init(
        project="attentivefp-spectra",
        config={
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_timesteps": args.num_timesteps,
        }
    )

    config = wandb.config

    # -----------------------
    # Data
    # -----------------------
    train_dataset = HierarchicalStreamingSpectraDataset(root=args.root_train)
    val_dataset = HierarchicalStreamingSpectraDataset(root=args.root_val)
    test_dataset = HierarchicalStreamingSpectraDataset(root=args.root_test)

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

    # -----------------------
    # Model
    # -----------------------
    # model = AttentiveFPGraphRegressor(
    #     node_feat_dim=train_dataset[0].x.shape[1],
    #     edge_feat_dim=train_dataset[0].edge_attr.shape[1],
    #     hidden_dim=args.hidden_dim,
    #     num_layers=args.num_layers,
    #     num_timesteps=args.num_timesteps,
    #     out_dim=174
    # )

    #model = BaselineGAT(
    #    node_feat_dim=train_dataset[0].x.shape[1],
    #   edge_feat_dim=train_dataset[0].edge_attr.shape[1],
    #    hidden_dim=args.hidden_dim,
    #    num_layers=args.num_layers,
    #    out_dim=174
    #)

    model = Hierachical_GAT(
         node_feat_dim=train_dataset[0].x.shape[1],
         edge_feat_dim=train_dataset[0].edge_attr.shape[1],
         hidden_dim=args.hidden_dim,
         num_layers=args.num_layers,
         out_dim=174
     )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used:', device)

    model = model.to(device)

    print('Model loaded.')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print('Optimizer loaded.')

    # save path dir creation
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # -----------------------
    # Step-based training loop
    # -----------------------
    best_loss = float('inf')

    print('Starting Training...')

    pbar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in pbar:

        data = next(train_iter)

        train_loss, batch_size = train_step(data)

        pbar.set_postfix(train_loss=train_loss)

        # -----------------------
        # Logging
        # -----------------------
        wandb.log({
            "step": step,
            "train_loss": train_loss
        })

        # -----------------------
        # Validation
        # -----------------------
        if step % config.eval_every == 0:

            val_loss = evaluate(val_loader, split="val")

            print(
                f"\nStep {step:06d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            wandb.log({
                "step": step,
                "val_loss": val_loss
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
    print("Loading best model...")

    model.load_state_dict(torch.load(args.save_path))

    test_loss = evaluate(test_loader, split="test")

    print("Test Loss:", test_loss)

    wandb.log({
        "test_loss": test_loss
    })

    wandb.finish()
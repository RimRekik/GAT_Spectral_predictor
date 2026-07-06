import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import os
from data.hierarchical_streaming_dataset import HierarchicalStreamingSpectraDataset
from model.model import build_model
from model.losses import masked_spectral_distance
from config import load_args


def infinite_loader(loader):
    """Wraps a DataLoader into an infinite iterator, reshuffling chunks every pass."""
    while True:
        loader.dataset.chunk_shuffle()
        for batch in loader:
            yield batch


def train_step(data):
    model.train()
    data = data.to(device)
    optimizer.zero_grad()
    out  = model(data)
    loss = masked_spectral_distance(data.y.view(data.num_graphs, -1), out)
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
        out  = model(data)
        loss = masked_spectral_distance(data.y.view(data.num_graphs, -1),out)
        total_loss += loss.item() * data.num_graphs
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader.dataset)


if __name__ == '__main__':

    args = load_args()

    # ── WandB ────────────────────────────────────────────────────────────────
    os.environ["WANDB_API_KEY"] = 'b4a27ac6b6145e1a5d0ee7f9e2e8c20bd101dccd'
    os.environ["WANDB_MODE"]    = "offline"
    os.environ["WANDB_DIR"]     = os.path.abspath("./wandb_run")

    wandb.init(
        project="attentivefp-spectra",
        config=vars(args),   # logs every arg including --model
    )
    config = wandb.config

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset = HierarchicalStreamingSpectraDataset(root=args.root_train)
    val_dataset   = HierarchicalStreamingSpectraDataset(root=args.root_val)
    test_dataset  = HierarchicalStreamingSpectraDataset(root=args.root_test)

    sample = train_dataset[0]
    print('Data loaded.')
    print(f"  node features : {sample.x.shape}")
    print(f"  edge features : {sample.edge_attr.shape}")
    print(f"  target        : {sample.y.shape}")

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=1, pin_memory=True)
    train_loader  = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader    = DataLoader(val_dataset,   **loader_kwargs)
    test_loader   = DataLoader(test_dataset,  **loader_kwargs)
    train_iter    = infinite_loader(train_loader)

    # ── Model — built from config.model ──────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    model = build_model({
        **vars(args),
        "node_feat_dim": sample.x.shape[1],
        "edge_feat_dim": sample.edge_attr.shape[1],
        "out_dim":       174,
    }).to(device)

    print(f"Model  : {args.model}  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps, eta_min=1e-5
    )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # ── Training loop (step-based) ────────────────────────────────────────────
    best_loss = float('inf')
    print('Starting training...')

    for step in (pbar := tqdm(range(1, config.max_steps + 1), desc="Training")):

        data = next(train_iter)
        train_loss, _ = train_step(data)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(train_loss=f"{train_loss:.4f}", lr=f"{current_lr:.2e}")

        wandb.log({"step": step, "train_loss": train_loss, "lr": current_lr})

        # ── Validation ───────────────────────────────────────────────────────
        if step % config.eval_every == 0:
            val_loss = evaluate(val_loader, split="val")

            print(f"\nStep {step:06d} | Train {train_loss:.4f} | Val {val_loss:.4f}")
            wandb.log({"step": step, "val_loss": val_loss})

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), args.save_path)
                print(f"  → best model saved (val_loss={best_loss:.4f})")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(args.save_path, weights_only=True))

    test_loss = evaluate(test_loader, split="test")
    print(f"Test loss : {test_loss:.4f}")

    wandb.log({"test_loss": test_loss})
    wandb.finish()
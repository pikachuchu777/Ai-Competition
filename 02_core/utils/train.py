import os
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from torch_geometric.loader import NeighborLoader

from models.model import GraphTransformer

# Config
SEED = 1018
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE    = 8192
EPOCHS        = 300
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
NUM_NEIGHBORS = [20, 20, 20]
N_SPLITS      = 5


def build_class_weight(y):
    """Build class weights for imbalanced binary classification."""
    pos_count = int((y == 1).sum())
    neg_count = int((y == 0).sum())
    return torch.tensor([1.0, neg_count / max(pos_count, 1)], device=DEVICE)


def build_dataloaders(data, train_nodes, val_nodes):
    """Create NeighborLoader instances for train and validation nodes."""
    train_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=train_nodes,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=val_nodes,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer):
    """Run a single training epoch over the neighbor-sampled batches."""
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr)
        logits = out[:batch.batch_size]
        mask = (batch.y[:batch.batch_size] != -1)
        if mask.sum() == 0:
            continue

        loss = criterion(logits[mask], batch.y[:batch.batch_size][mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def evaluate(model, val_loader):
    """Evaluate model on validation loader and compute PR-AUC / best F1."""
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            logits = out[:batch.batch_size]
            mask = (batch.y[:batch.batch_size] != -1)
            if mask.sum() == 0:
                continue

            prob = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
            y_true.append(batch.y[:batch.batch_size][mask].cpu().numpy())
            y_prob.append(prob)

    if not y_true:
        return 0.0, 0.0  # ap, f1

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    ap = average_precision_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    f1 = f1s.max() if len(f1s) else 0.0

    return ap, f1


# Main
def train_main():
    """Run class-weighted 5-fold training of GraphTransformer and save fold checkpoints."""
    print("[Train] Loading preprocessed graph ...")
    bundle = torch.load("cache/preprocessed_graph.pt", map_location="cpu", weights_only=False)
    data = bundle["data"]
    y = data.y

    # Loss function with class weights
    weight = build_class_weight(y)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Only use labeled nodes for CV
    labeled_idx = np.where(y.numpy() != -1)[0]
    labeled_y = y[labeled_idx].numpy()

    print(f"[Train] Starting {N_SPLITS}-Fold Stratified Cross-Validation ...\n")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_metrics = []
    os.makedirs("cache", exist_ok=True)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(labeled_idx, labeled_y), 1):
        print(f"===== Fold {fold}/{N_SPLITS} =====")
        train_nodes = torch.tensor(labeled_idx[tr_idx], dtype=torch.long)
        val_nodes   = torch.tensor(labeled_idx[val_idx], dtype=torch.long)

        train_loader, val_loader = build_dataloaders(data, train_nodes, val_nodes)

        model = GraphTransformer(
            in_dim=data.num_features,
            edge_in_dim=data.edge_attr.size(1)
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        best_f1 = 0.0
        best_state = None
        patience = 80
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            total_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            ap, f1 = evaluate(model, val_loader)

            print(
                f"Fold {fold:02d} Epoch {epoch:03d} | "
                f"Loss={total_loss:.4f} | PR-AUC={ap:.4f} | F1={f1:.4f}"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            # if patience_counter >= patience:
            #     print(f"[Train] Early stopping at epoch {epoch}")
            #     break

        if best_state is not None:
            state_path = os.path.join("cache", f"fold_{fold}.pt")
            torch.save(best_state, state_path)
            print(f"[Train] Saved best state for fold {fold} to {state_path}")

        fold_metrics.append(best_f1)
        print(f"Fold {fold} Best F1 = {best_f1:.4f}\n")

    print(f"[Train] Average F1 across {N_SPLITS} folds: {np.mean(fold_metrics):.4f}")


if __name__ == "__main__":
    train_main()

import os
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import amp
from torch.amp import GradScaler
from Model.model import Classifier

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


SEED = 42
HIDDEN_DIM = 100
NUM_LAYERS = 3
DROPOUT = 0.3

EPOCHS = 2880
PATIENCE = 100
LR = 3e-3
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "Cache/preprocessed_graph.pt"
ALERT_1 = pd.read_csv("Dataset/acct_alert.csv")
ALERT_2 = pd.read_csv("Dataset/phase2_acct_alert.csv")
ALERT   = pd.concat([ALERT_1, ALERT_2], axis=0, ignore_index=True)
PREDICT = pd.read_csv("Dataset/phase2_acct_predict.csv")


def make_seed(seed):
    '''
    Set Python, NumPy, and PyTorch random seeds.
    Ensures reproducibility across the entire training pipeline.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_cosine_warmup_scheduler(optimizer, num_epochs=EPOCHS, warmup_ratio=0.05):
    '''
    Build a cosine learning-rate scheduler with warmup.

    Args:
        optimizer     : PyTorch optimizer.
        num_epochs    : Total training epochs.
        warmup_ratio  : Ratio of epochs for warmup stage.

    Returns:
        lr_scheduler  : LambdaLR scheduler applying warmup + cosine decay.
    '''
    warmup_epochs = max(1, int(num_epochs * warmup_ratio))

    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(warmup_epochs)
        # Cosine decay
        progress = (current_epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def search_threshold(train_label, train_prob, val_label, val_prob):
    '''
    Search optimal classification threshold based on Validation F1 score

    Args:
        train_label, train_prob : Training ground truth + predicted prob
        val_label, val_prob     : Validation ground truth + predicted prob

    Returns:
        best_threshold : Best threshold satisfying recall constraint
        best_f1_score  : Corresponding validation F1 score
    '''
    threshold_grid = np.linspace(0.3, 0.7, 200)
    best_threshold = 0.5
    best_f1_score = -1.0

    for threshold in threshold_grid:
        recall_ok = True
        for y_true, p_prob in ((train_label, train_prob), (val_label, val_prob)):
            pred = (p_prob >= threshold).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall < 0.35:
                recall_ok = False
                break

        if not recall_ok:
            continue

        # Compute validation F1
        val_pred = (val_prob >= threshold).astype(int)
        tp = ((val_pred == 1) & (val_label == 1)).sum()
        fp = ((val_pred == 1) & (val_label == 0)).sum()
        fn = ((val_pred == 0) & (val_label == 1)).sum()
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else 0.0

        if f1 > best_f1_score:
            best_f1_score = f1
            best_threshold = threshold

    return best_threshold, best_f1_score


def train_main():
    '''
    Full training pipeline:
      - load graph
      - build features/labels
      - train GNN with early stop
      - search best threshold
      - save model + threshold
    '''
    make_seed(SEED)

    os.makedirs("Cache", exist_ok=True)

    obj = torch.load(PATH, map_location="cpu", weights_only=False)
    feat_df = obj["feat_df"].copy()
    edge_index = obj["edge_index"]
    mapping = obj["mapping"]

    feat_df["acct"] = feat_df["acct"].astype(str)

    # Column names
    alert_acct_col   = mapping.get("alert_acct", "acct")
    predict_acct_col = mapping.get("predict_acct", "acct")

    alert_accounts   = set(ALERT[alert_acct_col].astype(str).tolist())
    predict_accounts = set(PREDICT[predict_acct_col].astype(str).tolist())
    predict_list     = PREDICT[predict_acct_col].astype(str).tolist()

    labeled_accounts_df = feat_df[
        (feat_df["is_esun"] == 1) & (~feat_df["acct"].isin(predict_accounts))
    ].copy()

    y_train = labeled_accounts_df["acct"].map(
        lambda a: 1 if a in alert_accounts else 0
    ).astype(int).values

    all_accounts = pd.Index(feat_df["acct"].astype(str).unique())
    account_id_map = {acct: idx for idx, acct in enumerate(all_accounts)}

    feature_columns = [c for c in feat_df.columns if c != "acct"]
    X = (
        feat_df.set_index("acct")
        .loc[all_accounts][feature_columns]
        .astype(np.float32)
        .values
    )

    train_node_indices = np.array(
        [account_id_map[acct] for acct in labeled_accounts_df["acct"].astype(str)],
        dtype=np.int64
    )

    scaler = StandardScaler()
    X[train_node_indices] = scaler.fit_transform(X[train_node_indices])
    mean  = scaler.mean_
    scale = getattr(scaler, "scale_", (scaler.var_ + 1e-9) ** 0.5)

    mask = ~np.isin(np.arange(len(X)), train_node_indices)
    X[mask] = np.clip((X[mask] - mean) / (scale + 1e-12), -5, 5)

    alert_set   = set(alert_accounts)
    predict_set = set(predict_accounts)

    y_all = np.fromiter(
        (1.0 if acct in alert_set and acct not in predict_set else 0.0
         for acct in all_accounts),
        dtype=np.float32,
        count=len(all_accounts),
    )

    data = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx_raw, val_idx_raw = next(data.split(train_node_indices, y_train))

    train_nodes = train_node_indices[train_idx_raw]
    val_nodes   = train_node_indices[val_idx_raw]

    X_t    = torch.from_numpy(X).to(DEVICE)
    y_t    = torch.from_numpy(y_all).to(DEVICE)
    edge_t = edge_index.to(DEVICE)
    train_t = torch.from_numpy(train_nodes).long().to(DEVICE)
    val_t   = torch.from_numpy(val_nodes).long().to(DEVICE)

    model = Classifier(X.shape[1], HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)

    with torch.no_grad():
        num_pos = y_t[train_t].sum().item()
    num_neg = len(train_nodes) - num_pos
    pos_w = math.sqrt(max(num_neg / max(num_pos, 1), 1.0))
    pos_weight = torch.tensor([pos_w], dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = build_cosine_warmup_scheduler(optimizer, num_epochs=EPOCHS)
    scaler = GradScaler(enabled=True)

    val_label = y_t[val_t].cpu().numpy().astype(int)
    best_f1 = -1
    best_state = None
    no_improve = 0

    for epoch in tqdm(range(1, EPOCHS + 1), desc="Training", ncols=80):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        with amp.autocast("cuda", dtype=torch.float16):
            logit = model(X_t, edge_t)[train_t]
            label = y_t[train_t].float()

            eps = 0.02
            label = label * (1 - eps) + 0.5 * eps

            loss = F.binary_cross_entropy_with_logits(
                logit,
                label,
                pos_weight=pos_weight
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad(), amp.autocast("cuda", dtype=torch.float16):
                val_logit = model(X_t, edge_t)[val_t]

            val_prob = torch.sigmoid(val_logit).cpu().numpy()
            val_pred = (val_prob >= 0.5).astype(int)

            tp = ((val_pred == 1) & (val_label == 1)).sum()
            fp = ((val_pred == 1) & (val_label == 0)).sum()
            fn = ((val_pred == 0) & (val_label == 1)).sum()
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom > 0 else 0

            print(f"[Epoch {epoch:04d}] LR={optimizer.param_groups[0]['lr']:.6f} "
                  f"Loss={loss.item():.4f} F1@0.5={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= PATIENCE:
                print(f"[EarlyStop] best F1={best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, "Cache/best_model.pth")
    print("[Save] best model to Cache/best_model.pth")

    model = model.cpu().float()
    X_cpu   = torch.from_numpy(X).float()
    edge_cpu = edge_index.cpu()

    model.eval()
    with torch.no_grad():
        all_prob = torch.sigmoid(model(X_cpu, edge_cpu)).numpy()

    best_thr, thr_f1 = search_threshold(
        train_label=y_all[train_nodes].astype(int),
        train_prob=all_prob[train_nodes],
        val_label=y_all[val_nodes].astype(int),
        val_prob=all_prob[val_nodes],
    )

    print(f"[Threshold] best={best_thr:.4f}, F1={thr_f1:.4f}")

    val_pred_final = (all_prob[val_nodes] >= best_thr).astype(int)
    tp = ((val_pred_final == 1) & (val_label == 1)).sum()
    fp = ((val_pred_final == 1) & (val_label == 0)).sum()
    fn = ((val_pred_final == 0) & (val_label == 1)).sum()
    denom = 2 * tp + fp + fn
    final_f1 = 2 * tp / denom if denom > 0 else 0

    print(f"[Valid] thr={best_thr:.4f}, F1={final_f1:.4f}")

    with open("Cache/best_threshold.txt", "w") as f:
        f.write(str(best_thr))

    print("Save threshold to Cache/best_threshold.txt")


if __name__ == "__main__":
    train_main()

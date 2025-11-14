import os
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
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
NUM_NEIGHBORS = [20, 20, 20]


def load_preprocessed_graph():
    """Load the preprocessed graph bundle from disk."""
    print("[Infer] Loading preprocessed graph ...")
    bundle = torch.load("cache/preprocessed_graph.pt", map_location="cpu", weights_only=False)
    data = bundle["data"]
    acct_classes = bundle["acct_le_classes"]
    return data, acct_classes


def load_checkpoints(data):
    """Load all saved fold checkpoints and build corresponding models."""
    ckpts = sorted(glob.glob(os.path.join("cache", "fold_*.pt")))
    if not ckpts:
        raise FileNotFoundError("No fold_*.pt found in cache/.")

    print(f"[Infer] Found {len(ckpts)} checkpoints: {ckpts}")

    models = []
    for path in ckpts:
        model = GraphTransformer(
            in_dim=data.num_features,
            edge_in_dim=data.edge_attr.size(1)
        ).to(DEVICE)
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    return models


def build_infer_loader(data):
    """Create a NeighborLoader that iterates over all nodes for inference."""
    num_nodes = data.num_nodes
    acct_list = torch.arange(num_nodes, dtype=torch.long)
    infer_loader = NeighborLoader(
        data,
        num_neighbors=NUM_NEIGHBORS,
        input_nodes=acct_list,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return infer_loader, num_nodes


def run_inference(models, data):
    """Run ensemble inference over all nodes and return mean probabilities."""
    infer_loader, num_nodes = build_infer_loader(data)

    probs_list = []
    for mid, model in enumerate(models):
        print(f"[Infer] Inferring with model {mid+1}/{len(models)} ...")
        probs_fold = np.zeros(num_nodes, dtype=np.float32)

        with torch.no_grad():
            for batch in tqdm(infer_loader, desc=f"Infer Fold {mid+1}", leave=False):
                batch = batch.to(DEVICE)
                out = model(batch.x, batch.edge_index, batch.edge_attr)
                prob = F.softmax(out[:batch.batch_size], dim=1)[:, 1]
                center_nid = batch.n_id[:batch.batch_size]
                probs_fold[center_nid.cpu().numpy()] = prob.detach().cpu().numpy()

        probs_list.append(probs_fold)

    mean_probs = np.mean(probs_list, axis=0)
    return mean_probs


def build_submission(acct_classes, mean_probs, thr=0.5):
    """Build submission DataFrame and write it to CSV."""
    print("[Infer] Building submission.csv ...")
    predict = pd.read_csv("01_dataset/acct_predict.csv")

    pred_df = pd.DataFrame({"acct": acct_classes, "prob": mean_probs})
    sub = predict.merge(pred_df, on="acct", how="left").fillna({"prob": 0})
    sub["label"] = (sub["prob"] >= thr).astype(int)

    os.makedirs("03_results", exist_ok=True)
    out_path = "03_results/submission.csv"
    sub[["acct", "label"]].to_csv(out_path, index=False)
    print(f"[Infer] Saved to {out_path}")


# Main
def inference_main():
    """Run ensemble inference using saved checkpoints and generate submission.csv."""
    data, acct_classes = load_preprocessed_graph()
    models = load_checkpoints(data)
    mean_probs = run_inference(models, data)
    build_submission(acct_classes, mean_probs, thr=0.5)


if __name__ == "__main__":
    inference_main()

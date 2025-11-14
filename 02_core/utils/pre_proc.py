import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
tqdm.pandas()

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy


# Global Config
SEED = 1018
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOPK_CURRENCY   = 8
EDGE_EMB_DIM    = 32
PATH_DIM        = 64
WALK_LEN        = 5
NUM_PATHS       = 10
TOP_K_FEATURES  = 20


# Helper Functions
def to_hour(s):
    """Convert 'HH:MM:SS' strings to continuous hour values."""
    t = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
    h = t.dt.hour.fillna(0) + t.dt.minute.fillna(0) / 60 + t.dt.second.fillna(0) / 3600
    return h.fillna(12.0)


def safe_autocorr(x):
    """Compute lag-1 autocorrelation safely (return 0 for degenerate cases)."""
    x = np.asarray(x)
    if x.size < 2 or np.all(x == x[0]):
        return 0.0
    s = pd.Series(x).autocorr(lag=1)
    return float(s) if pd.notnull(s) else 0.0


def entropy_of(series):
    """Compute Shannon entropy (base 2) of a categorical distribution."""
    p = series.value_counts(normalize=True)
    return float(entropy(p, base=2)) if len(p) else 0.0


def hhi(series):
    """Compute Herfindahl–Hirschman Index (HHI) for a categorical distribution."""
    if series.empty:
        return 0.0
    p = series.value_counts(normalize=True).values
    return float(np.sum(p ** 2))


def granular_bucket(h):
    """Map hour-of-day to a granular time bucket label."""
    if 0 <= h < 6:
        return "EarlyMorning_0_6"
    if 6 <= h < 9:
        return "Morning_6_9"
    if 9 <= h < 12:
        return "LateMorning_9_12"
    if 12 <= h < 14:
        return "Noon_12_14"
    if 14 <= h < 18:
        return "Afternoon_14_18"
    if 18 <= h < 21:
        return "Evening_18_21"
    return "LateNight_21_24"


def safe_ratio(a, b):
    """Compute a/(b+eps) with safe handling of zeros."""
    return np.where((a + b) > 0, a / (b + 1e-9), 0.0)


def per_acct_temporal(sub):
    """Build simple temporal features for a single account."""
    sub = sub.sort_values("txn_date")
    return pd.Series(
        {"autocorr_txn_amt_lag1": safe_autocorr(sub["txn_amt"].to_numpy())}
    )


def robust_amount_anom(sub, min_len=20, z_thr=3.5):
    """Detect robust amount anomalies using MAD-based z-score."""
    if len(sub) < min_len:
        return pd.Series({"robust_anom_cnt": 0.0, "robust_anom_posmean": 0.0})
    v = np.log1p(pd.to_numeric(sub["txn_amt"], errors="coerce").fillna(0.0).to_numpy())
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    z = 0.6745 * (v - med) / mad
    z_pos = np.clip(z, 0.0, None)
    return pd.Series(
        {
            "robust_anom_cnt": float((z > z_thr).sum()),
            "robust_anom_posmean": float(z_pos.mean()),
        }
    )


def build_neighbors_numpy(edge_index, num_nodes):
    """Build adjacency lists (neighbors) from edge indices."""
    ei = edge_index.cpu().numpy()
    nbrs = [[] for _ in range(num_nodes)]
    for u, v in zip(ei[0], ei[1]):
        nbrs[u].append(v)
    out = []
    for n in nbrs:
        if n:
            out.append(np.asarray(n, dtype=np.int64))
        else:
            out.append(np.empty((0,), dtype=np.int64))
    return out


def random_walk_numpy(neighbors, starts, walk_length):
    """Perform uniform random walks of a fixed length."""
    starts = np.asarray(starts, dtype=np.int64)
    Np = starts.shape[0]
    walks = np.empty((Np, walk_length + 1), dtype=np.int64)
    walks[:, 0] = starts
    cur = starts.copy()
    rng = np.random.default_rng(SEED)
    for step in trange(1, walk_length + 1, desc="RandomWalk steps", leave=False):
        nxt = cur.copy()
        for i in range(Np):
            neigh = neighbors[cur[i]]
            if neigh.size > 0:
                nxt[i] = rng.choice(neigh)
        walks[:, step] = nxt
        cur = nxt
    return walks


def sample_walks_no_cluster(edge_index_undir, start_nodes, walk_length, num_paths_per_node):
    """Sample multiple random walks per node on an undirected graph."""
    neighbors = build_neighbors_numpy(edge_index_undir, num_nodes=int(edge_index_undir.max().item()) + 1)
    starts = start_nodes.cpu().numpy().repeat(num_paths_per_node)
    walks_np = random_walk_numpy(neighbors, starts, walk_length)
    return torch.from_numpy(walks_np).to(start_nodes.device)


class PathEncoder(nn.Module):
    """GRU-based encoder that maps random walks to fixed-size embeddings."""

    def __init__(self, num_levels, d_model=64, path_len=6):
        """
        Initialize the path encoder.

        Args:
            num_levels: Number of discrete node levels.
            d_model: Embedding dimension.
            path_len: Length of each random walk (excluding the start).
        """
        super().__init__()
        self.node_tok = nn.Embedding(num_levels, d_model)
        self.pos = nn.Parameter(torch.randn(path_len + 1, d_model))
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, walks_levels):
        """
        Encode random-walk level sequences into path embeddings.

        Args:
            walks_levels: Tensor of shape [num_paths, path_len+1]
                containing node level indices.

        Returns:
            Tensor of shape [num_paths, d_model] with path embeddings.
        """
        x = self.node_tok(walks_levels)
        x = x + self.pos[: x.size(1)]
        y, _ = self.gru(x)
        return self.out_proj(y[:, -1])


@torch.no_grad()
def build_path_context(edge_index_undir, node_level, walk_len, num_paths, d_model, device):
    """Build per-node path context embeddings via random walks and PathEncoder."""
    num_nodes = int(node_level.numel())
    start_nodes = torch.arange(num_nodes, device=device)
    walks = sample_walks_no_cluster(edge_index_undir.to(device), start_nodes, walk_len, num_paths)
    levels = node_level.to(device)[walks]
    enc = PathEncoder(
        num_levels=int(node_level.max().item()) + 1,
        d_model=d_model,
        path_len=walk_len,
    ).to(device)
    enc.eval()

    B = 262144
    all_vecs = []
    for i in trange(0, levels.size(0), B, desc="Encode path context", leave=False):
        all_vecs.append(enc(levels[i : i + B]))
    path_vec = torch.cat(all_vecs, dim=0)

    center = start_nodes.repeat_interleave(num_paths)
    path_ctx = torch.zeros(num_nodes, d_model, device=device)
    path_cnt = torch.zeros(num_nodes, 1, device=device)
    path_ctx.index_add_(0, center, path_vec)
    path_cnt.index_add_(0, center, torch.ones_like(center, dtype=torch.float32).unsqueeze(-1))
    return path_ctx / path_cnt.clamp_min(1.0)


def load_and_prepare_transactions(txn_path, alert_path):
    """Load raw CSVs and perform basic transaction cleaning and time features."""
    print("[Preprocess] Reading data ...")
    txn = pd.read_csv(txn_path).iloc[:-1]
    alert = pd.read_csv(alert_path)

    txn["txn_amt"] = pd.to_numeric(txn["txn_amt"], errors="coerce").fillna(0.0)
    txn["txn_date"] = pd.to_numeric(txn["txn_date"], errors="coerce").fillna(0).astype(int)
    for c in [
        "from_acct_type",
        "to_acct_type",
        "from_acct",
        "to_acct",
        "currency_type",
        "channel_type",
        "is_self_txn",
    ]:
        txn[c] = txn[c].astype(str)

    txn["is_self_num"] = txn["is_self_txn"].str.upper().eq("Y").astype(int)
    txn["txn_hour"] = to_hour(txn["txn_time"]).astype(float)

    txn["is_night"] = ((txn["txn_hour"] >= 22) | (txn["txn_hour"] < 6)).astype(int)
    txn["is_weekend"] = ((txn["txn_date"] % 7).isin([0, 6])).astype(int)

    return txn, alert


def build_account_features(txn, alert):
    """Construct account-level behavioral features from transaction data."""
    print("[Preprocess] Generating account-level behavior features ...")

    alert_set = set(alert["acct"].astype(str).unique())

    base = (
        txn.groupby("from_acct")
        .agg(
            count_total_transactions=("txn_amt", "count"),
            sum_txn_amt=("txn_amt", "sum"),
            mean_txn_amt=("txn_amt", "mean"),
            std_txn_amt=("txn_amt", "std"),
            max_txn_amt=("txn_amt", "max"),
            min_txn_amt=("txn_amt", "min"),
            count_unique_to_acct=("to_acct", pd.Series.nunique),
            count_unique_channel=("channel_type", pd.Series.nunique),
            min_txn_date=("txn_date", "min"),
            max_txn_date=("txn_date", "max"),
            active_days=("txn_date", pd.Series.nunique),
            mean_txn_hour=("txn_hour", "mean"),
            std_txn_hour=("txn_hour", "std"),
        )
        .fillna(0.0)
    )
    base["range_txn_date"] = (base["max_txn_date"] - base["min_txn_date"]).fillna(0)
    base["txn_freq_per_day"] = np.where(
        base["active_days"] > 0,
        base["count_total_transactions"] / base["active_days"],
        0.0,
    )

    # Self / other transactions
    g_self = txn[txn["is_self_num"] == 1].groupby("from_acct")["txn_amt"].agg(
        count_self_transactions="count",
        sum_txn_amt_self="sum",
        mean_txn_amt_self="mean",
        std_txn_amt_self="std",
        max_txn_amt_self="max",
    )
    g_other = txn[txn["is_self_num"] == 0].groupby("from_acct")["txn_amt"].agg(
        count_other_transactions="count",
        sum_txn_amt_other="sum",
        mean_txn_amt_other="mean",
        std_txn_amt_other="std",
        max_txn_amt_other="max",
    )
    self_ratio = txn.groupby("from_acct")["is_self_num"].mean().rename("self_ratio").fillna(0.0)

    # In / out structure
    out_nonself = (
        txn[txn["is_self_num"] == 0]
        .groupby("from_acct")["to_acct"]
        .count()
        .rename("outgoing_nonself_cnt")
    )
    in_nonself = (
        txn[txn["is_self_num"] == 0]
        .groupby("to_acct")["from_acct"]
        .count()
        .rename("incoming_nonself_cnt")
    )

    out_nonself_amt = (
        txn[txn["is_self_num"] == 0]
        .groupby("from_acct")["txn_amt"]
        .sum()
        .rename("outgoing_nonself_amt")
    )
    in_nonself_amt = (
        txn[txn["is_self_num"] == 0]
        .groupby("to_acct")["txn_amt"]
        .sum()
        .rename("incoming_nonself_amt")
    )

    out_unique = (
        txn[txn["is_self_num"] == 0]
        .groupby("from_acct")["to_acct"]
        .nunique()
        .rename("outgoing_unique_to")
    )
    in_unique = (
        txn[txn["is_self_num"] == 0]
        .groupby("to_acct")["from_acct"]
        .nunique()
        .rename("incoming_unique_from")
    )

    out_hhi = (
        txn[txn["is_self_num"] == 0]
        .groupby("from_acct")["to_acct"]
        .apply(hhi)
        .rename("outgoing_hhi")
    )
    in_hhi = (
        txn[txn["is_self_num"] == 0]
        .groupby("to_acct")["from_acct"]
        .apply(hhi)
        .rename("incoming_hhi")
    )

    # Alert-alert edges
    aa_edges = txn[
        (txn["is_self_num"] == 0)
        & (txn["from_acct"].isin(alert_set))
        & (txn["to_acct"].isin(alert_set))
    ]
    aa_out_cnt = aa_edges.groupby("from_acct").size().rename("alert2alert_out_cnt")
    aa_in_cnt = aa_edges.groupby("to_acct").size().rename("alert2alert_in_cnt")

    # Channel distribution
    ch_tab = txn.groupby(["from_acct", "channel_type"]).size().unstack(fill_value=0)
    ch_ratio = ch_tab.div(ch_tab.sum(axis=1), axis=0).add_prefix("channel_ratio::")
    entropy_channel = (
        txn.groupby("from_acct")["channel_type"]
        .progress_apply(entropy_of)
        .rename("channel_entropy")
    )

    # Granular time buckets
    txn["gran_bucket"] = txn["txn_hour"].apply(granular_bucket)
    gr_tab = txn.groupby(["from_acct", "gran_bucket"]).size().unstack(fill_value=0)
    gr_ratio = gr_tab.div(gr_tab.sum(axis=1), axis=0).add_prefix("gran_ratio::")

    night_ratio = txn.groupby("from_acct")["is_night"].mean().rename("night_ratio")
    weekend_ratio = txn.groupby("from_acct")["is_weekend"].mean().rename("weekend_ratio")

    mean_hour = txn.groupby("from_acct")["txn_hour"].mean().rename("mean_hour").fillna(0.0)
    sin_hour = np.sin(2 * np.pi * mean_hour / 24).rename("sin_hour")
    cos_hour = np.cos(2 * np.pi * mean_hour / 24).rename("cos_hour")

    afternoon_ratio = gr_ratio.get(
        "gran_ratio::Afternoon_14_18",
        pd.Series(0.0, index=gr_ratio.index),
    ).rename("afternoon_ratio")

    close_to_14 = ((txn["txn_hour"] >= 13.5) & (txn["txn_hour"] < 14.5)).astype(int)
    peak14_ratio = (
        txn.assign(peak14=close_to_14)
        .groupby("from_acct")["peak14"]
        .mean()
        .rename("peak14_ratio")
        .fillna(0.0)
    )

    # Temporal features
    temporal_feat = txn.groupby("from_acct").progress_apply(per_acct_temporal)
    robust_anom = txn.groupby("from_acct").progress_apply(robust_amount_anom)

    # Merge everything
    node_feat = (
        base.join(
            [
                g_self,
                g_other,
                self_ratio,
                out_nonself,
                in_nonself,
                out_nonself_amt,
                in_nonself_amt,
                out_unique,
                in_unique,
                out_hhi,
                in_hhi,
                aa_out_cnt,
                aa_in_cnt,
                ch_ratio,
                entropy_channel,
                gr_ratio,
                night_ratio,
                weekend_ratio,
                mean_hour,
                sin_hour,
                cos_hour,
                afternoon_ratio,
                peak14_ratio,
                temporal_feat,
                robust_anom,
            ]
        )
        .fillna(0.0)
        .reset_index()
        .rename(columns={"from_acct": "acct"})
    )

    # Derived ratios / roles
    node_feat["inout_cnt_ratio"] = safe_ratio(
        node_feat["incoming_nonself_cnt"], node_feat["outgoing_nonself_cnt"]
    )
    node_feat["inout_amt_ratio"] = safe_ratio(
        node_feat["incoming_nonself_amt"], node_feat["outgoing_nonself_amt"]
    )
    node_feat["in_minus_out_cnt"] = (
        node_feat["incoming_nonself_cnt"] - node_feat["outgoing_nonself_cnt"]
    )
    node_feat["in_minus_out_amt"] = (
        node_feat["incoming_nonself_amt"] - node_feat["outgoing_nonself_amt"]
    )

    node_feat["has_incoming"] = (node_feat["incoming_nonself_cnt"] > 0).astype(int)
    node_feat["has_outgoing"] = (node_feat["outgoing_nonself_cnt"] > 0).astype(int)
    node_feat["role_only_in"] = (
        (node_feat["has_incoming"] == 1) & (node_feat["has_outgoing"] == 0)
    ).astype(int)
    node_feat["role_only_out"] = (
        (node_feat["has_incoming"] == 0) & (node_feat["has_outgoing"] == 1)
    ).astype(int)
    node_feat["role_bidir"] = (
        (node_feat["has_incoming"] == 1) & (node_feat["has_outgoing"] == 1)
    ).astype(int)

    for col in node_feat.columns:
        if str(node_feat[col].dtype)[0] in "fc":
            node_feat[col] = node_feat[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    print(
        f"[Preprocess] Generated {node_feat.shape[1]} features "
        f"for {len(node_feat)} accounts (currency features removed)."
    )
    return node_feat


def select_top_k_features(node_feat, alert, k):
    """Compute feature importance with RandomForest and keep top-k features."""
    print("[Preprocess] Calculating feature importances with RandomForest ...")

    label_df = alert[["acct"]].copy()
    label_df["label"] = 1
    node_lab_full = (
        node_feat.merge(label_df, on="acct", how="left")
        .fillna({"label": 0})
        .astype({"label": int})
    )

    X_imp = node_lab_full.drop(columns=["acct", "label"])
    y_imp = node_lab_full["label"].values

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=SEED,
        class_weight="balanced_subsample",
    )
    rf.fit(X_imp, y_imp)

    imp_df = (
        pd.DataFrame(
            {"feature": X_imp.columns, "importance": rf.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    selected_cols = imp_df.head(k)["feature"].tolist()
    print(f"[Preprocess] Top-{k} features selected:")
    print(imp_df.head(k).to_string(index=False))

    node_feat_reduced = node_feat[["acct"] + selected_cols]
    node_lab = (
        node_feat_reduced.merge(label_df, on="acct", how="left")
        .fillna({"label": 0})
        .astype({"label": int})
    )

    return node_feat_reduced, node_lab, selected_cols


def build_graph_data(txn, node_feat_reduced, node_lab):
    """Construct the PyG Data object: node features, edges, edge attributes, labels."""
    # Map accounts to node indices
    all_accts = np.unique(txn[["from_acct", "to_acct"]].values.flatten())
    acct_le = LabelEncoder().fit(all_accts)
    acct2id = {a: i for i, a in enumerate(acct_le.classes_)}

    src = txn["from_acct"].map(acct2id).values
    dst = txn["to_acct"].map(acct2id).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Edge features
    max_date = txn["txn_date"].max()
    txn["time_decay"] = np.exp(-(max_date - txn["txn_date"]) / 30.0)

    ch_le = LabelEncoder()
    txn["channel_type"] = txn["channel_type"].fillna("UNK")
    txn["ch_id"] = ch_le.fit_transform(txn["channel_type"])
    Kc = len(ch_le.classes_)

    top_c = txn["currency_type"].value_counts().head(TOPK_CURRENCY).index
    txn["cur_type"] = txn["currency_type"].apply(lambda x: x if x in top_c else "OTHER")
    cur_le = LabelEncoder()
    txn["cur_id"] = cur_le.fit_transform(txn["cur_type"])
    Ku = len(cur_le.classes_)

    txn["log_amt"] = np.log1p(txn["txn_amt"])
    amt_scaler = StandardScaler().fit(txn[["log_amt"]])
    txn["amt_z"] = amt_scaler.transform(txn[["log_amt"]])

    edge_raw = np.concatenate(
        [
            txn[["amt_z", "is_self_num", "time_decay"]].values.astype(np.float32),
            np.eye(Kc, dtype=np.float32)[txn["ch_id"].values],
            np.eye(Ku, dtype=np.float32)[txn["cur_id"].values],
        ],
        axis=1,
    ).astype(np.float32)
    edge_attr = torch.tensor(edge_raw, dtype=torch.float32)

    # Node feature matrix
    feat_all = (
        pd.DataFrame({"acct": acct_le.classes_})
        .merge(node_feat_reduced, on="acct", how="left")
        .fillna(0.0)
    )
    X = feat_all.drop(columns=["acct"]).values.astype(np.float32)
    x_scaler = StandardScaler().fit(X)
    x = torch.tensor(x_scaler.transform(X), dtype=torch.float32)

    # Labels
    y_map = node_lab.set_index("acct")["label"].to_dict()
    y = torch.tensor([y_map.get(a, -1) for a in acct_le.classes_], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    train_idx = torch.where(y != -1)[0]

    return data, acct_le.classes_, train_idx, txn


def compute_node_levels(txn, acct_classes):
    """Compute per-account discrete levels from frequency/degree statistics."""
    deg_out = txn.groupby("from_acct").size().rename("deg_out")
    deg_in = txn.groupby("to_acct").size().rename("deg_in")
    days = txn.groupby("from_acct")["txn_date"].nunique().rename("active_days")
    freq = (
        pd.concat([deg_out, deg_in, days], axis=1)
        .reindex(acct_classes)
        .fillna(0)
    )
    freq["freq_score"] = (
        freq["deg_out"] * 0.6
        + freq["deg_in"] * 0.3
        + freq["active_days"] * 0.1
    )
    LEVELS = 6
    ranked = pd.Series(freq["freq_score"].rank(method="first"), index=freq.index)
    freq["level"] = pd.qcut(
        ranked,
        q=min(LEVELS, ranked.nunique()),
        labels=False,
        duplicates="drop",
    ).fillna(0).astype(int)
    node_level = torch.tensor(freq["level"].values, dtype=torch.long)
    return node_level


def attach_path_context(data, txn, acct_classes):
    """Build path context embeddings and concatenate them to node features."""
    print("[Preprocess] Building path context ...")

    node_level = compute_node_levels(txn, acct_classes)

    edge_index_undir = to_undirected(data.edge_index)
    edge_index_undir = torch.unique(edge_index_undir.t(), dim=0).t()

    with torch.no_grad():
        path_ctx = build_path_context(
            edge_index_undir,
            node_level,
            WALK_LEN,
            NUM_PATHS,
            PATH_DIM,
            DEVICE,
        ).cpu()

    data.x = torch.cat([data.x, path_ctx], dim=1)
    return data


def save_preprocessed_graph(data, train_idx, acct_classes, selected_cols, out_path="cache/preprocessed_graph.pt"):
    """Save the preprocessed graph and metadata to disk."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "data": data,
            "train_idx": train_idx,
            "acct_le_classes": acct_classes,
            "selected_cols": selected_cols,
        },
        out_path,
    )
    print(f"[Preprocess] Saved preprocessed graph to {out_path}")


# Main
def main():
    """Run the full preprocessing pipeline and write preprocessed_graph.pt."""
    txn_path = "01_dataset/acct_transaction.csv"
    alert_path = "01_dataset/acct_alert.csv"

    txn, alert = load_and_prepare_transactions(txn_path, alert_path)
    node_feat = build_account_features(txn, alert)
    node_feat_reduced, node_lab, selected_cols = select_top_k_features(
        node_feat, alert, TOP_K_FEATURES
    )
    data, acct_classes, train_idx, txn = build_graph_data(
        txn, node_feat_reduced, node_lab
    )
    data = attach_path_context(data, txn, acct_classes)
    save_preprocessed_graph(data, train_idx, acct_classes, selected_cols)


if __name__ == "__main__":
    main()

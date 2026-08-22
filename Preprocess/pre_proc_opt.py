import os
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

SEED = 42


def make_seed(seed):
    '''
    Set all random seeds (Python, NumPy, PyTorch) to ensure reproducibility.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def auto_time_to_hms(t):
    '''
    Convert time column into (hour, minute, second).
    Supports both formats:
      - Numeric HHMMSS  (e.g., 93015 → 09:30:15)
      - String "HH:MM:SS" (e.g., "09:30:15")

    Returns:
        h, m, s  (pd.Series of int)
    '''
    if t.astype(str).str.contains(":").any():
        dt = pd.to_datetime(t, format="%H:%M:%S", errors="coerce")
        return (
            dt.dt.hour.fillna(0).astype(int),
            dt.dt.minute.fillna(0).astype(int),
            dt.dt.second.fillna(0).astype(int),
        )

    num = pd.to_numeric(t, errors="coerce")
    h = (num // 10000).clip(0, 23)
    m = ((num // 100) % 100).clip(0, 59)
    s = (num % 100).clip(0, 59)
    return h.fillna(0).astype(int), m.fillna(0).astype(int), s.fillna(0).astype(int)


def add_temporal_features(txn,
                          date_col="txn_date",
                          time_col="txn_time",
                          from_col="from_acct",
                          to_col="to_acct"):
    '''
    Add temporal-related features to transaction data:
        - pseudo timestamp & date
        - hour, weekday, night/weekend flags
        - time differences between consecutive transactions
        - rapid in/out indicators (< 60 seconds)

    Returns updated DataFrame with temporal features.
    '''
    if date_col not in txn.columns or time_col not in txn.columns:
        return txn

    tqdm.write("[Temporal] Adding temporal features ...")
    txn = txn.copy()

    d = pd.to_numeric(txn[date_col], errors="coerce")
    base = pd.Timestamp("2000-01-01")
    days_offset = (d - 1).fillna(0).clip(lower=0)
    base_date = base + pd.to_timedelta(days_offset, unit="D")

    # Parse time safely
    h, m, s = auto_time_to_hms(txn[time_col])
    seconds_in_day = ((h * 60 + m) * 60 + s).fillna(0)

    # Build pseudo timestamp
    pseudo_ts = base_date + pd.to_timedelta(seconds_in_day, unit="s")
    txn["pseudo_ts"] = pseudo_ts
    txn["pseudo_date"] = base_date.dt.date
    txn["hour"] = h.astype(int)
    txn["weekday"] = base_date.dt.weekday
    txn["is_night"] = txn["hour"].between(18, 24).astype(int)
    txn["is_weekend"] = txn["weekday"].isin([5, 6]).astype(int)
    txn["date"] = txn["pseudo_date"]

    # Outgoing time diff
    tqdm.write("[Temporal] Computing out-side time-diff ...")
    txn = txn.sort_values([from_col, "pseudo_ts"])
    txn["time_diff_out"] = (
        txn.groupby(from_col)["pseudo_ts"].diff().dt.total_seconds().fillna(0)
    )
    txn["is_rapid_out"] = (txn["time_diff_out"] < 60).astype(int)

    # Incoming time diff
    tqdm.write("[Temporal] Computing in-side time-diff ...")
    tmp = txn.sort_values([to_col, "pseudo_ts"]).copy()
    time_diff_in = (
        tmp.groupby(to_col)["pseudo_ts"].diff().dt.total_seconds().fillna(0)
    )
    txn["time_diff_in"] = time_diff_in.reindex(txn.index).fillna(0)
    txn["is_rapid_in"] = (txn["time_diff_in"] < 60).astype(int)

    return txn


def build_graph(txn, alert, predict, register=None):
    '''
    Build graph features and edge_index from transaction data.

    Returns:
        feat_df    (DataFrame of node features)
        edge_index (torch.LongTensor, shape [2, E])
        mapping    (column mapping dictionary)
        acct2idx   (account → node index dictionary)
    '''
    mapping = {
        "src": "from_acct",
        "dst": "to_acct",
        "amt": "txn_amt",
        "from_type": "from_acct_type",
        "to_type": "to_acct_type",
        "alert_acct": "acct",
        "predict_acct": "acct",
        "date": "txn_date",
        "time": "txn_time",
    }

    src_col = mapping["src"]
    dst_col = mapping["dst"]
    amt_col = mapping["amt"]

    tqdm.write("[Graph] Checking time feature availability ...")
    has_time = (mapping["date"] in txn.columns) and (mapping["time"] in txn.columns)

    if has_time:
        txn = add_temporal_features(
            txn,
            date_col=mapping["date"],
            time_col=mapping["time"],
            from_col=src_col,
            to_col=dst_col,
        )

    tqdm.write("[Graph] Aggregating outgoing features ...")
    outgoing = (
        txn.groupby(src_col)
        .agg(
            sum_out_amt=(amt_col, "sum"),
            max_out_amt=(amt_col, "max"),
            min_out_amt=(amt_col, "min"),
            avg_out_amt=(amt_col, "mean"),
            out_deg=(dst_col, "nunique"),
            out_txn_count=(dst_col, "size"),
        )
    )

    tqdm.write("[Graph] Aggregating incoming features ...")
    incoming = (
        txn.groupby(dst_col)
        .agg(
            sum_in_amt=(amt_col, "sum"),
            max_in_amt=(amt_col, "max"),
            min_in_amt=(amt_col, "min"),
            avg_in_amt=(amt_col, "mean"),
            in_deg=(src_col, "nunique"),
            in_txn_count=(src_col, "size"),
        )
    )

    if has_time:
        tqdm.write("[Graph] Temporal aggregation ...")
        time_agg_out = txn.groupby(src_col).agg(
            avg_time_diff_out=("time_diff_out", "mean"),
            rapid_ratio_out=("is_rapid_out", "mean"),
            night_ratio_out=("is_night", "mean"),
            weekend_ratio_out=("is_weekend", "mean"),
            active_days_out=("date", "nunique"),
            txn_per_day_out=("date", lambda x: len(x) / max(x.nunique(), 1)),
        )

        time_agg_in = txn.groupby(dst_col).agg(
            avg_time_diff_in=("time_diff_in", "mean"),
            rapid_ratio_in=("is_rapid_in", "mean"),
            night_ratio_in=("is_night", "mean"),
            weekend_ratio_in=("is_weekend", "mean"),
            active_days_in=("date", "nunique"),
            txn_per_day_in=("date", lambda x: len(x) / max(x.nunique(), 1)),
        )

        idx = (
            outgoing.index
            .union(incoming.index)
            .union(time_agg_out.index)
            .union(time_agg_in.index)
        )

        feat_df = (
            pd.DataFrame(index=idx)
            .join(outgoing, how="left")
            .join(incoming, how="left")
            .join(time_agg_out, how="left")
            .join(time_agg_in, how="left")
        )
    else:
        idx = outgoing.index.union(incoming.index)
        feat_df = (
            pd.DataFrame(index=idx)
            .join(outgoing, how="left")
            .join(incoming, how="left")
        )

    feat_df = feat_df.fillna(0.0).reset_index().rename(columns={"index": "acct"})
    feat_df["acct"] = feat_df["acct"].astype(str)

    tqdm.write("[Graph] Computing reciprocity & smurfing features ...")

    # Reciprocity
    pairs = txn[[src_col, dst_col]].astype(str).dropna()
    rev_pairs = pairs.rename(columns={src_col: "rev_src", dst_col: "rev_dst"})
    recip = pairs.merge(
        rev_pairs,
        left_on=[src_col, dst_col],
        right_on=["rev_dst", "rev_src"],
        how="inner",
    )
    recip_count_out = recip.groupby(src_col).size()
    recip_count_in = recip.groupby(dst_col).size()
    feat_df["recip_out"] = feat_df["acct"].map(recip_count_out).fillna(0)
    feat_df["recip_in"] = feat_df["acct"].map(recip_count_in).fillna(0)

    # Smurfing score
    out_txn_cnt = txn.groupby(src_col).size()
    out_partner_cnt = txn.groupby(src_col)[dst_col].nunique()
    out_dispersion = out_partner_cnt / out_txn_cnt.replace(0, np.nan)
    out_concentration = 1.0 / out_dispersion.replace(0, np.nan)

    in_txn_cnt = txn.groupby(dst_col).size()
    in_partner_cnt = txn.groupby(dst_col)[src_col].nunique()
    in_dispersion = in_partner_cnt / in_txn_cnt.replace(0, np.nan)

    out_conc_for_acct = out_concentration.reindex(feat_df["acct"]).astype(float)
    in_disp_for_acct = in_dispersion.reindex(feat_df["acct"]).astype(float)

    feat_df["smurfing_score"] = (
        (in_disp_for_acct * out_conc_for_acct)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    feat_df["is_esun"] = 1
    esun_set = set()
    for type_col, acct_col in (("from_acct_type", src_col), ("to_acct_type", dst_col)):
        if type_col in txn.columns:
            tmp = txn[[acct_col, type_col]].dropna().drop_duplicates()
            col_val = tmp[type_col].astype(str).str.strip()
            esun_mask = col_val.isin(["01", "1"])
            esun_set.update(tmp.loc[esun_mask, acct_col].astype(str).tolist())
    if esun_set:
        feat_df["is_esun"] = feat_df["acct"].isin(esun_set).astype(int)

    # Register features
    reg_cols = [
        "reg_out_cnt", "reg_out_final_active_cnt", "reg_out_long_cnt",
        "reg_out_total_days", "reg_out_avg_days",
        "reg_in_cnt", "reg_in_final_active_cnt", "reg_in_long_cnt",
        "reg_in_total_days", "reg_in_avg_days",
    ]

    if register is not None and len(register) > 0:
        tqdm.write("[Register] Aggregating improved register features ...")
        reg = register.copy()

        rf = "from_acct"
        rt = "to_acct"
        rs = "start_date"
        re_ = "end_date"

        reg[rf] = reg[rf].astype(str)
        reg[rt] = reg[rt].astype(str)

        start_raw = pd.to_numeric(reg[rs], errors="coerce")
        end_raw = pd.to_numeric(reg[re_], errors="coerce")

        if mapping["date"] in txn.columns:
            max_day = (
                pd.to_numeric(txn[mapping["date"]], errors="coerce")
                .fillna(1).astype(int).max()
            )
        else:
            max_day = 999

        start_eff = start_raw.fillna(-1).astype(int)
        end_eff = end_raw.fillna(999).astype(int)

        start_eff = np.where(start_eff < 1, 1, start_eff)
        end_eff = np.where(end_eff > max_day, max_day, end_eff)

        valid_mask = end_eff >= start_eff
        reg = reg.loc[valid_mask].copy()

        if reg.empty:
            feat_df = feat_df.set_index("acct")
            for c in reg_cols:
                if c not in feat_df.columns:
                    feat_df[c] = 0.0
            feat_df = feat_df.reset_index()
        else:
            reg["start_eff"] = start_eff[valid_mask]
            reg["end_eff"] = end_eff[valid_mask]
            reg["active_days"] = (reg["end_eff"] - reg["start_eff"]).clip(lower=0)

            reg["is_final_active"] = (reg["end_eff"] >= max_day).astype(int)
            reg["is_long_term"] = (
                end_raw.loc[valid_mask].fillna(999).astype(int) == 999
            ).astype(int)

            reg_out = (
                reg.groupby(rf)
                .agg(
                    reg_out_cnt=(rt, "size"),
                    reg_out_final_active_cnt=("is_final_active", "sum"),
                    reg_out_long_cnt=("is_long_term", "sum"),
                    reg_out_total_days=("active_days", "sum"),
                    reg_out_avg_days=("active_days", "mean"),
                )
            )

            reg_in = (
                reg.groupby(rt)
                .agg(
                    reg_in_cnt=(rf, "size"),
                    reg_in_final_active_cnt=("is_final_active", "sum"),
                    reg_in_long_cnt=("is_long_term", "sum"),
                    reg_in_total_days=("active_days", "sum"),
                    reg_in_avg_days=("active_days", "mean"),
                )
            )

            feat_df = (
                feat_df.set_index("acct")
                .join(reg_out, how="left")
                .join(reg_in, how="left")
                .fillna(0.0)
                .reset_index()
            )
    else:
        feat_df = feat_df.set_index("acct")
        for c in reg_cols:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        feat_df = feat_df.reset_index()

    feat_col = [
        "sum_out_amt", "sum_in_amt",
        "max_out_amt", "min_out_amt", "avg_out_amt",
        "max_in_amt", "min_in_amt", "avg_in_amt",
        "out_txn_count", "in_txn_count",
        "recip_out", "recip_in", "smurfing_score",
    ] + reg_cols

    if has_time:
        feat_col += [
            "avg_time_diff_out", "avg_time_diff_in",
            "active_days_out", "active_days_in",
            "txn_per_day_out", "txn_per_day_in",
        ]

    for col in tqdm(feat_col, desc="log1p features"):
        if col in feat_df.columns:
            feat_df["log1p_" + col] = np.log1p(
                pd.to_numeric(feat_df[col], errors="coerce").fillna(0.0)
            )

    temp = [
        "acct", "is_esun",
        "sum_out_amt", "sum_in_amt",
        "max_out_amt", "min_out_amt", "avg_out_amt",
        "max_in_amt", "min_in_amt", "avg_in_amt",
        "out_deg", "in_deg",
        "out_txn_count", "in_txn_count",
        "recip_out", "recip_in",
        # register features
        "reg_out_cnt", "reg_out_final_active_cnt", "reg_out_long_cnt",
        "reg_out_total_days", "reg_out_avg_days",
        "reg_in_cnt", "reg_in_final_active_cnt", "reg_in_long_cnt",
        "reg_in_total_days", "reg_in_avg_days",
    ]

    if has_time:
        temp += [
            "avg_time_diff_out", "rapid_ratio_out",
            "night_ratio_out", "weekend_ratio_out",
            "active_days_out", "txn_per_day_out",
            "avg_time_diff_in", "rapid_ratio_in",
            "night_ratio_in", "weekend_ratio_in",
            "active_days_in", "txn_per_day_in",
        ]

    temp += [col for col in feat_df.columns if col.startswith("log1p_")]
    temp = [c for c in temp if c in feat_df.columns]

    feat_df = feat_df[temp].fillna(0.0)

    tqdm.write("[Graph] Building edge_index ...")

    acct_list = feat_df["acct"].tolist()
    acct2idx = {a: i for i, a in enumerate(acct_list)}

    edges = txn[[src_col, dst_col]].dropna().astype(str)
    src_idx = edges[src_col].map(acct2idx)
    dst_idx = edges[dst_col].map(acct2idx)
    mask = src_idx.notna() & dst_idx.notna()

    if not mask.any():
        edge_index = torch.empty(2, 0, dtype=torch.long)
    else:
        src = src_idx[mask].astype(np.int64).to_numpy()
        dst = dst_idx[mask].astype(np.int64).to_numpy()
        edge_arr = np.vstack([src, dst])
        edge_arr = np.unique(edge_arr, axis=1)
        src, dst = edge_arr

        u = np.concatenate([src, dst])
        v = np.concatenate([dst, src])
        edge_index = torch.from_numpy(np.vstack([u, v])).long()

    tqdm.write("[Graph] Graph build complete!")

    return feat_df, edge_index, mapping, acct2idx


def save_preprocessed_graph(
    txn_path_1,
    alert_path_1,
    txn_path_2,
    alert_path_2,
    predict_path,
    register_path,
    out_path
):
    '''
    Load CSV files, build graph, and save the processed result to disk.

    Saves:
        - feat_df     (node features)
        - edge_index  (graph edges)
        - mapping     (column mapping)
        - acct2idx    (node id mapping)
    '''
    make_seed(SEED)


    tqdm.write("[IO] Loading CSV ...")
    txn_df_1   = pd.read_csv(txn_path_1)
    alert_df_1 = pd.read_csv(alert_path_1)
    txn_df_2   = pd.read_csv(txn_path_2)
    alert_df_2 = pd.read_csv(alert_path_2)
    
    txn_df = pd.concat([txn_df_1, txn_df_2], axis=0, ignore_index=True)
    alert_df = pd.concat([alert_df_1, alert_df_2], axis=0, ignore_index=True)

    predict_df = pd.read_csv(predict_path)

    register_df = None
    if register_path is not None and os.path.exists(register_path):
        tqdm.write(f"[IO] Loading register CSV from {register_path} ...")
        register_df = pd.read_csv(register_path)
    else:
        tqdm.write("[IO] Register CSV not found or path None, skip register features.")

    tqdm.write("[Graph] Starting graph preprocessing ...")
    feat_df, edge_index, mapping, acct2idx = build_graph(
        txn_df, alert_df, predict_df, register_df
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    obj = {
        "feat_df": feat_df,
        "edge_index": edge_index,
        "mapping": mapping,
        "acct2idx": acct2idx,
    }

    torch.save(obj, out_path)
    tqdm.write(f"[Save] Preprocessed graph saved: {out_path}")
    tqdm.write(f"Nodes = {len(feat_df)}, Edges = {edge_index.size(1)}")

def main():
    '''
    Main entry point for preprocessing the transaction graph.
    '''

    save_preprocessed_graph(
        txn_path_1="Dataset/acct_transaction.csv",
        alert_path_1="Dataset/acct_alert.csv",

        txn_path_2="Dataset/phase2_acct_transaction.csv",
        alert_path_2="Dataset/phase2_acct_alert.csv",
        
        predict_path="Dataset/phase2_acct_predict.csv",
        register_path="Dataset/phase2_acct_register.csv",
        out_path="Cache/preprocessed_graph.pt"
    )


if __name__ == "__main__":
    main()

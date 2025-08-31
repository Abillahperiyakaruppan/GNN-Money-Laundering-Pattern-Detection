# dataset.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def load_graph_from_csv(trans_path="transactions.csv", acc_path="accounts.csv"):
    tx = pd.read_csv(trans_path)
    acc = pd.read_csv(acc_path)

    accounts = sorted(acc['account_id'].unique())
    id2idx = {aid: i for i, aid in enumerate(accounts)}
    n_nodes = len(accounts)

    # edges
    src = tx['src'].map(id2idx).values
    dst = tx['dst'].map(id2idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # edge attributes: log(amount) and hour-of-day
    amounts = np.log1p(tx['amount'].values).astype(np.float32)
    timestamps = pd.to_datetime(tx['timestamp'])
    hours = timestamps.dt.hour.astype(np.float32).values
    edge_attr = torch.tensor(np.vstack([amounts, hours]).T, dtype=torch.float)

    # node features: signup_days, in/out degree, log total in/out
    out_deg = np.zeros(n_nodes, dtype=float)
    in_deg = np.zeros(n_nodes, dtype=float)
    total_in = np.zeros(n_nodes, dtype=float)
    total_out = np.zeros(n_nodes, dtype=float)

    for s,d,a in zip(src, dst, tx['amount'].values):
        out_deg[s] += 1
        in_deg[d] += 1
        total_out[s] += a
        total_in[d] += a

    signup = acc.set_index('account_id')['signup_days'].to_dict()
    X = np.vstack([
        [signup[a] for a in accounts],
        in_deg,
        out_deg,
        np.log1p(total_in),
        np.log1p(total_out)
    ]).T.astype(np.float32)

    x = torch.tensor(X, dtype=torch.float)

    # labels
    is_susp = acc.set_index('account_id')['is_suspicious'].to_dict()
    y = torch.tensor([is_susp.get(a, 0) for a in accounts], dtype=torch.float).unsqueeze(1)

    # random masks split
    n = n_nodes
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_n = int(0.7 * n)
    val_n = int(0.15 * n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:train_n]] = True
    val_mask[idx[train_n:train_n+val_n]] = True
    test_mask[idx[train_n+val_n:]] = True

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

if __name__ == "__main__":
    d = load_graph_from_csv()
    print(d)

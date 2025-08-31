# inference.py
import torch
import pandas as pd
from dataset import load_graph_from_csv
from model import GCNNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph_from_csv()
    model = GCNNet(in_channels=data.num_node_features, hidden_channels=128, dropout=0.0).to(device)
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
    data.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()

    accounts = pd.read_csv("accounts.csv")['account_id'].tolist()
    out = pd.DataFrame({"account_id": accounts, "suspicious_score": probs})
    out = out.sort_values("suspicious_score", ascending=False)
    out.to_csv("scores.csv", index=False)
    print("Saved scores.csv - top suspicious accounts:")
    print(out.head(20))
# train.py
import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from dataset import load_graph_from_csv
from model import GCNNet
import numpy as np

def evaluate(model, data, mask, device):
    model.eval()
    with torch.no_grad():
        logits = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    y_true = data.y.cpu().numpy().flatten()
    mask_idx = mask.cpu().numpy()
    if mask_idx.sum() == 0:
        return {"precision":0,"recall":0,"f1":0,"auc":float("nan")}
    y_true_masked = y_true[mask_idx]
    y_scores_masked = probs[mask_idx]
    y_pred_masked = (y_scores_masked >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true_masked, y_pred_masked, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(y_true_masked, y_scores_masked)
    except:
        auc = float('nan')
    return {"precision": p, "recall": r, "f1": f1, "auc": auc}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_graph_from_csv()
    data.to(device)

    model = GCNNet(in_channels=data.num_node_features, hidden_channels=128, dropout=0.4).to(device)

    # class imbalance handling
    y = data.y.cpu().numpy().flatten()
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    pos_weight = torch.tensor([neg / pos], dtype=torch.float).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    best_val_f1 = -1
    save_path = "best_model.pt"
    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].squeeze(1))
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            train_met = evaluate(model, data, data.train_mask, device)
            val_met = evaluate(model, data, data.val_mask, device)
            print(f"Epoch {epoch:03d} Loss {loss.item():.4f} TrainF1 {train_met['f1']:.3f} ValF1 {val_met['f1']:.3f} ValAUC {val_met['auc']:.3f}")
            if val_met['f1'] > best_val_f1:
                best_val_f1 = val_met['f1']
                torch.save(model.state_dict(), save_path)
                print("  Saved best model")

    # test
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_met = evaluate(model, data, data.test_mask, device)
    print("Test metrics:", test_met)

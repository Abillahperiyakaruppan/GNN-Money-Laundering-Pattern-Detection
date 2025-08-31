Project goal:
    Detecting the suspicious transaction pattern.

Dataset Required :
- Transactions dataset with:  
  transaction_id, sender_id, receiver_id, amount, time, location, transaction_type
- Labelled as suspicious or normal


a. Data Preprocessing
- Clean missing/invalid entries
- Normalize amounts, time, and frequency

b. Graph Construction (for GNN)
- Nodes = accounts
- Edges = transactions
- Features = transaction metadata

c. Model Development
- Use GNN (e.g., GCN, GAT) or traditional ML (Random Forest, XGBoost)
- Train on historical data with labels

d. Detection
- Classify new/unseen transactions as suspicious or normal
- Output alerts or flag IDs

e. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix to visualize results

Tech Stack :
- Python
- PyTorch Geometric (for GNN)
- Pandas, NetworkX, Scikit-learn
- Matplotlib/Seaborn for visualization
"""Two-node dimuon relational GNN prototype (NOT a general HEP event graph).

Each event is a 2-node graph (muon1 <-> muon2) with 8 features per muon
(E, px, py, pz, pt, eta, phi, Q) and DeltaR-weighted edges. Since the
four-momenta determine the invariant mass M, and labels are 80<M<100,
this network learns a known kinematic relation — a useful graph-ML
exercise, not a discovery architecture.

Protocol: stratified train/val/test, seeded, AUROC/AUPRC, checkpointed.
Default uses a 10k-row subset for speed; use --full for the whole file and
do NOT compare subset numbers against full-data XGBoost numbers.
"""
import argparse
import os
import random
import pandas as pd
import numpy as np
import torch
try:
    from torch_geometric.loader import DataLoader
except ImportError:  # backward compat with old PyG
    from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

print("🚀 Loading PyTorch Geometric Framework for Dimuon Graph Construction...")

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.fc = torch.nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def convert_to_graph_dataset(df):
    """Transforms tabular dimuon rows into 2-node PyTorch Geometric graphs."""
    dataset = []
    for _, row in df.iterrows():
        node1 = [row['E1'], row['px1'], row['py1'], row['pz1'], row['pt1'], row['eta1'], row['phi1'], row['Q1']]
        node2 = [row['E2'], row['px2'], row['py2'], row['pz2'], row['pt2'], row['eta2'], row['phi2'], row['Q2']]
        x = torch.tensor([node1, node2], dtype=torch.float)

        d_eta = row['eta1'] - row['eta2']
        d_phi = row['phi1'] - row['phi2']
        if d_phi > np.pi:
            d_phi -= 2 * np.pi
        if d_phi < -np.pi:
            d_phi += 2 * np.pi

        delta_r = np.sqrt(d_eta**2 + d_phi**2)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        weight = np.exp(-delta_r)
        edge_weight = torch.tensor([weight, weight], dtype=torch.float)

        label = 1 if 80 < row['M'] < 100 else 0
        y = torch.tensor([label], dtype=torch.long)

        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y))
    return dataset


def load_graph_data(filepath="Dimuon_DoubleMu.root", nrows=10000):
    """Loads ROOT binary into stratified train/val/test PyG DataLoaders."""
    import uproot
    import awkward as ak
    try:
        file = uproot.open(filepath)
    except FileNotFoundError:
        print("❌ ROOT dataset not found. Run src/data_download.py.")
        raise
    data = file["Events"].arrays()
    if hasattr(data, 'to_dataframe'):
        df = data.to_dataframe()
    else:
        df = ak.to_dataframe(data)

    df = df.dropna()
    if nrows is not None:
        df = df.iloc[:nrows]
        print(f"⚠️ GNN subset mode: using first {len(df)} rows only — not comparable to full-data XGBoost.")
    else:
        print(f"✅ Loaded {len(df)} collision events (full file).")
    print(f"✅ Loaded {len(df)} discrete collision events from TTree format.")

    graph_dataset = convert_to_graph_dataset(df)
    labels = [int(d.y.item()) for d in graph_dataset]
    train_ds, temp_ds = train_test_split(
        graph_dataset, test_size=0.4, random_state=SEED, stratify=labels)
    temp_labels = [int(d.y.item()) for d in temp_ds]
    val_ds, test_ds = train_test_split(
        temp_ds, test_size=0.5, random_state=SEED, stratify=temp_labels)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader


def train_gnn(train_loader, val_loader=None, device='cpu', epochs=10, lr=0.01):
    """Standard training loop with validation tracking."""
    model = GCN(num_node_features=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"🚀 Commencing PyTorch Node-Edge Training on {device}...")

    best_val_loss = float('inf')
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(len(train_loader), 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            vl = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    vl += F.nll_loss(model(data), data.y).item()
            val_loss = vl / max(len(val_loader), 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        history.append((train_loss, val_loss))
        if epoch % 2 == 0:
            msg = f"   Epoch {epoch:03d} | train loss: {train_loss:.4f}"
            if val_loss is not None:
                msg += f" | val loss: {val_loss:.4f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        print(f"✅ Restored best-val checkpoint (val loss {best_val_loss:.4f}).")
    return model, history


def evaluate_gnn(model, test_loader, device='cpu'):
    """Reports accuracy + AUROC/AUPRC on the frozen test split."""
    print("\n⚖️ Evaluating Graph Classification (frozen TEST)...")
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            logp = model(data)
            prob = logp.exp()[:, 1]
            pred = logp.max(dim=1)[1]
            all_probs.extend(prob.cpu().numpy().tolist())
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(data.y.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auroc, auprc = float('nan'), float('nan')
    print(f"🎯 GNN TEST Accuracy: {acc*100:.2f}% | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    return {"accuracy": acc, "auroc": auroc, "auprc": auprc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='use full dataset instead of 10k subset')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    set_seeds(SEED)
    # NO blanket try/except: real failures must surface, not exit 0 silently.
    train_loader, val_loader, test_loader = load_graph_data(
        nrows=None if args.full else 10000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_gnn(train_loader, val_loader, device=device, epochs=args.epochs)
    metrics = evaluate_gnn(model, test_loader, device=device)

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics,
        'seed': SEED,
        'val_split': True,
    }, 'models/gnn_prototype.pt')
    print("💾 Geometric prototype (+optimizer-free checkpoint) saved as models/gnn_prototype.pt")


if __name__ == '__main__':
    main()

"""Two-node dimuon relational GNN prototype (NOT a general HEP event graph).

Each event is a 2-node graph (muon1 <-> muon2) with 8 features per muon
(E, px, py, pz, pt, eta, phi, Q) and DeltaR-weighted edges. Since the
four-momenta determine the invariant mass M, and labels are 80<M<100,
this network learns a known kinematic relation — a useful graph-ML
exercise, not a discovery architecture.

Protocol: stratified train/val/test, seeded, AUROC/AUPRC, checkpointed.
Default uses a random 10k-row subset (seeded) for speed; use --full for the
whole file and do NOT compare subset numbers against full-data XGBoost numbers.
"""
import argparse
import os
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import data_path, ROOT

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
GRAPH_COLUMNS = ['E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1',
                 'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2', 'M']


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


def convert_to_graph_dataset(df, scaler=None):
    """Transforms tabular dimuon rows into 2-node PyTorch Geometric graphs (vectorized)."""
    missing = set(GRAPH_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Graph data is missing required columns {sorted(missing)}")

    cols1 = ['E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1']
    cols2 = ['E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2']
    n1 = df[cols1].to_numpy(dtype=np.float32)
    n2 = df[cols2].to_numpy(dtype=np.float32)
    nodes = np.stack([n1, n2], axis=1)

    if scaler is not None:
        orig_shape = nodes.shape
        nodes = scaler.transform(nodes.reshape(-1, orig_shape[-1])).reshape(orig_shape).astype(np.float32)

    d_eta = df['eta1'].to_numpy(dtype=np.float32) - df['eta2'].to_numpy(dtype=np.float32)
    d_phi = df['phi1'].to_numpy(dtype=np.float32) - df['phi2'].to_numpy(dtype=np.float32)
    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
    delta_r = np.sqrt(d_eta**2 + d_phi**2)
    weights = np.exp(-delta_r)

    labels = ((df['M'].to_numpy() > 80) & (df['M'].to_numpy() < 100)).astype(np.int64)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    dataset = []
    for i in range(len(df)):
        x = torch.from_numpy(nodes[i])
        w = float(weights[i])
        edge_weight = torch.tensor([w, w], dtype=torch.float)
        y = torch.tensor([labels[i]], dtype=torch.long)
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y))
    return dataset


def load_graph_data(filepath="Dimuon_DoubleMu.root", nrows=10000, batch_size=64):
    """Loads ROOT binary into stratified train/val/test PyG DataLoaders with leakage-free scaling."""
    import uproot
    import awkward as ak
    from sklearn.preprocessing import StandardScaler
    try:
        with uproot.open(data_path(filepath)) as file:
            data = file["Events"].arrays()
            if hasattr(data, 'to_dataframe'):
                df = data.to_dataframe()
            else:
                df = ak.to_dataframe(data)
    except FileNotFoundError:
        print("❌ ROOT dataset not found. Run src/data_download.py.")
        raise

    df = df.dropna()
    if nrows is not None and len(df) > nrows:
        # Seeded random subset (NOT first-N: file order is not guaranteed
        # representative, so iloc[:nrows] could bias the signal fraction).
        df = df.sample(n=nrows, random_state=SEED).reset_index(drop=True)
        print(f"⚠️ GNN subset mode: using {len(df)} random rows (seed {SEED}) — not comparable to full-data XGBoost.")
    else:
        print(f"✅ Loaded {len(df)} collision events (full file).")

    labels = ((df['M'] > 80) & (df['M'] < 100)).astype(int).values
    train_df, temp_df = train_test_split(
        df, test_size=0.4, random_state=SEED, stratify=labels)
    temp_labels = ((temp_df['M'] > 80) & (temp_df['M'] < 100)).astype(int).values
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_labels)

    # Fit feature scaler strictly on TRAIN node features to avoid test data leakage
    cols1 = ['E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1']
    cols2 = ['E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2']
    train_nodes = np.vstack([train_df[cols1].to_numpy(dtype=np.float32), train_df[cols2].to_numpy(dtype=np.float32)])
    scaler = StandardScaler().fit(train_nodes)

    train_ds = convert_to_graph_dataset(train_df, scaler=scaler)
    val_ds = convert_to_graph_dataset(val_df, scaler=scaler)
    test_ds = convert_to_graph_dataset(test_df, scaler=scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scaler


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
        if epoch == 1 or epoch == epochs or epoch % 2 == 0:
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
    parser.add_argument('--batch-size', type=int, default=None, help='batch size (default: 128 for full, 64 for subset)')
    args = parser.parse_args()

    set_seeds(SEED)
    bs = args.batch_size if args.batch_size is not None else (128 if args.full else 64)
    # NO blanket try/except: real failures must surface, not exit 0 silently.
    train_loader, val_loader, test_loader, scaler = load_graph_data(
        nrows=None if args.full else 10000, batch_size=bs)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_gnn(train_loader, val_loader, device=device, epochs=args.epochs)
    metrics = evaluate_gnn(model, test_loader, device=device)

    model_dir = ROOT / 'models'
    os.makedirs(model_dir, exist_ok=True)
    save_path = str(model_dir / 'gnn_prototype.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'metrics': metrics,
        'scaler': scaler,
        'seed': SEED,
        'val_split': True,
    }, save_path)
    print(f"💾 Geometric prototype (+optimizer-free checkpoint) saved as {save_path}")


if __name__ == '__main__':
    main()

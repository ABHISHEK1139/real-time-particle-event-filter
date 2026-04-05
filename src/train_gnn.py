import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

print("🚀 Loading PyTorch Geometric Framework for Dimuon Graph Construction...")

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
    """
    Transforms tabular dimuon rows into PyTorch Geometric structural graphs.
    """
    dataset = []
    
    for _, row in df.iterrows():
        node1 = [row['E1'], row['px1'], row['py1'], row['pz1'], row['pt1'], row['eta1'], row['phi1'], row['Q1']]
        node2 = [row['E2'], row['px2'], row['py2'], row['pz2'], row['pt2'], row['eta2'], row['phi2'], row['Q2']]
        x = torch.tensor([node1, node2], dtype=torch.float)
        
        d_eta = row['eta1'] - row['eta2']
        d_phi = row['phi1'] - row['phi2']
        if d_phi > np.pi: d_phi -= 2 * np.pi
        if d_phi < -np.pi: d_phi += 2 * np.pi
            
        delta_r = np.sqrt(d_eta**2 + d_phi**2)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        
        weight = np.exp(-delta_r)
        edge_weight = torch.tensor([weight, weight], dtype=torch.float)
        
        label = 1 if 80 < row['M'] < 100 else 0
        y = torch.tensor([label], dtype=torch.long)
        
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y))
        
    return dataset

def load_graph_data(filepath="Dimuon_DoubleMu.root", nrows=10000):
    """Loads a small sample from ROOT binary into PyTorch Geometric DataLoaders."""
    try:
        import uproot
        import awkward as ak
        file = uproot.open(filepath)
        data = file["Events"].arrays()
        if hasattr(data, 'to_dataframe'):
            df = data.to_dataframe()
        else:
            df = ak.to_dataframe(data)
            
        df = df.dropna()[:nrows]
        print(f"✅ Loaded {len(df)} discrete collision events from TTree format.")
        
        graph_dataset = convert_to_graph_dataset(df)
        train_dataset, test_dataset = train_test_split(graph_dataset, test_size=0.2, random_state=42)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        return train_loader, test_loader
        
    except FileNotFoundError:
        print("❌ ROOT dataset not found. Run src/data_download.py.")
        raise

def train_gnn(train_loader, device='cpu'):
    """Executes the standard PyTorch training loop on the topology structures."""
    model = GCN(num_node_features=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    print(f"🚀 Commencing PyTorch Node-Edge Training on {device}...")
    
    for epoch in range(1, 11):
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 2 == 0:
            print(f"   Epoch {epoch:03d} | Progressive Loss: {total_loss/len(train_loader):.4f}")
    return model

def evaluate_gnn(model, test_loader, device='cpu'):
    """Performs inference generation tracking accuracy without explicitly tracking mass matrices."""
    print("\n⚖️ Evaluating Graph Classification Accuracy...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data).max(dim=1)[1]
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"🎯 Proof-of-Concept GNN Accuracy: {acc*100:.2f}%")
    return acc

def main():
    try:
        train_loader, test_loader = load_graph_data()
    except Exception as e:
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_gnn(train_loader, device=device)
    evaluate_gnn(model, test_loader, device=device)
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/gnn_prototype.pt')
    print("💾 Geometric prototype saved as models/gnn_prototype.pt")

if __name__ == '__main__':
    main()

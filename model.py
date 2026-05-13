import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.data import Batch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ------------------------
# Config
# ------------------------
split = 1
num_epochs = 30
bs = 32
lr = 1e-4
pretrained_path = f"./weights/bigsol_cross_attention1.pt"
best_model_path = f"./weights/water_solubility_model.pt"
data_dir = './data_files'

# ----------------------------------------
# Helper functions
# ----------------------------------------
def transform(y, eps=1e-12):
    return np.log10(y + eps)

def inverse_transform(y_log):
    return 10 ** y_log


# ----------------------------------------
# Temperature Modulator
# ----------------------------------------
class TempModulator(nn.Module):
    def __init__(self, emb_dim, temp_stats_path="./data_files/bigsol_temperatureK.pkl", n_rbf=10, gamma=None):
        super().__init__()
        # Load temperature statistics
        with open(temp_stats_path, "rb") as f:
            temps = pickle.load(f)
        if isinstance(temps, dict):
            temps = np.array(list(temps.values()))
        temps = np.array(temps, dtype=np.float32)

        # Register mean and std
        t_mean = np.mean(temps)
        t_std = np.std(temps)
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float))
        self.register_buffer("t_std", torch.tensor(t_std, dtype=torch.float))

        # Set RBF centers based on actual stats (z-scores from -3 to +3)
        z_min, z_max = -3.0, 3.0
        centers = np.linspace(z_min, z_max, n_rbf)
        self.register_buffer("rbf_centers", torch.tensor(centers, dtype=torch.float))

        # Gamma
        if gamma is None:
            gamma = 1.0 / (2 * ((z_max - z_min) / (n_rbf - 1)) ** 2)  # spacing-based gamma
        self.gamma = gamma

        # FC layer
        self.fc = nn.Sequential(
            nn.Linear(n_rbf, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid()
        )


    def rbf_expand(self, t_norm):
        diff = t_norm.unsqueeze(-1) - self.rbf_centers
        return torch.exp(-self.gamma * diff.pow(2))

    def forward(self, embedding, temperature):
        t_norm = (temperature - self.t_mean) / (self.t_std + 1e-8)
        t_rbf = self.rbf_expand(t_norm)
        t_emb = self.fc(t_rbf.to(embedding.device))
        return embedding * t_emb


# -----------------------------
# Deep Mixture Network (DMN)
# -----------------------------
class DMNHead(nn.Module):
    def __init__(self, input_dim, n_components=3):
        super().__init__()
        self.n_components = n_components
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), int(input_dim/4)),
        )
        self.mu = nn.Linear(int(input_dim/4), n_components)
        self.log_sigma = nn.Linear(int(input_dim/4), n_components)
        self.pi_logits = nn.Linear(int(input_dim/4), n_components)

    def forward(self, x):
        h = self.fc(x)
        mu = self.mu(h)
        sigma = torch.exp(self.log_sigma(h))
        pi = F.softmax(self.pi_logits(h), dim=-1)
        return mu, sigma, pi


# -----------------------------
# DMN Loss (Negative Log-Likelihood)
# -----------------------------
def dmn_loss(y_true, mu, sigma, pi):
    y_true = y_true.unsqueeze(-1)
    const = torch.sqrt(torch.tensor(2.0 * torch.pi, device=y_true.device))
    prob = (1.0 / (const * sigma)) * torch.exp(-0.5 * ((y_true - mu) / sigma) ** 2)
    weighted_prob = (pi * prob).sum(dim=-1) + 1e-12
    nll = -torch.log(weighted_prob)
    return nll.mean()


# -----------------------------
# Cross-attention layer
# -----------------------------
class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv):
        if x_q.size(0) == 0 or x_kv.size(0) == 0:
            return torch.zeros_like(x_q)

        Nq, D = x_q.shape
        Nk, _ = x_kv.shape
        H, hd = self.num_heads, self.head_dim

        q = self.q_proj(x_q).view(Nq, H, hd).permute(1, 0, 2)
        k = self.k_proj(x_kv).view(Nk, H, hd).permute(1, 2, 0)
        v = self.v_proj(x_kv).view(Nk, H, hd).permute(1, 0, 2)

        attn = F.softmax(torch.matmul(q, k) / (hd ** 0.5), dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).permute(1, 0, 2).contiguous().view(Nq, D)
        return self.out_proj(out)


# -----------------------------
# Main Model
# -----------------------------
class SolubilityPredictor(nn.Module):
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=128, n_components=3, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.solute_embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.solvent_embed = nn.Linear(atom_feat_dim, hidden_dim)

        self.cross_attn_solute = CrossAttentionLayer(hidden_dim, num_heads=num_heads)
        self.cross_attn_solvent = CrossAttentionLayer(hidden_dim, num_heads=num_heads)

        self.solute_encoder = AttentiveFP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=bond_feat_dim,
            num_layers=2,
            num_timesteps=2
        )
        self.solvent_encoder = AttentiveFP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=bond_feat_dim,
            num_layers=1,
            num_timesteps=1
        )

        self.temp_mod_solvent = TempModulator(hidden_dim)
        self.dmn = DMNHead(input_dim=hidden_dim * 2, n_components=n_components)

    def _per_sample_cross_attention(self, solute_x, solute_batch, solvent_x, solvent_batch):
        device = solute_x.device
        num_graphs = int(solute_batch.max().item()) + 1
        solute_out = torch.zeros_like(solute_x)
        solvent_out = torch.zeros_like(solvent_x)

        for i in range(num_graphs):
            s_mask, v_mask = (solute_batch == i), (solvent_batch == i)
            if s_mask.sum() == 0 or v_mask.sum() == 0:
                continue
            s_idx, v_idx = torch.nonzero(s_mask).squeeze(-1), torch.nonzero(v_mask).squeeze(-1)
            s_nodes, v_nodes = solute_x[s_idx], solvent_x[v_idx]
            solute_out[s_idx] = s_nodes + self.cross_attn_solute(s_nodes, v_nodes)
            solvent_out[v_idx] = v_nodes + self.cross_attn_solvent(v_nodes, s_nodes)
        return solute_out, solvent_out

    def forward(self, solute_data, solvent_data, temperature):
        solute_x = self.solute_embed(solute_data.x)
        solvent_x = self.solvent_embed(solvent_data.x)
        solute_x_cross, solvent_x_cross = self._per_sample_cross_attention(
            solute_x, solute_data.batch, solvent_x, solvent_data.batch
        )

        h_solute = self.solute_encoder(solute_x_cross, solute_data.edge_index, solute_data.edge_attr, batch=solute_data.batch)
        h_solvent = self.solvent_encoder(solvent_x_cross, solvent_data.edge_index, solvent_data.edge_attr, batch=solvent_data.batch)
        h_solvent_mod = self.temp_mod_solvent(h_solvent, temperature)
        h = torch.cat([h_solute, h_solvent_mod], dim=-1)
        mu, sigma, pi = self.dmn(h)
        return mu, sigma, pi


# -----------------------------
# Dataset + Collate
# -----------------------------
class SolubilityDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        try:
            solute, solvent, temp, _, label = self.data_list[idx]
        except:
            solute, solvent, temp, label = self.data_list[idx]
        # label = transform(label)
        return solute, solvent, torch.tensor(temp, dtype=torch.float), torch.tensor(label, dtype=torch.float)

def solubility_collate_fn(batch):
    solutes, solvents, temps, labels = zip(*batch)
    solute_batch = Batch.from_data_list(list(solutes))
    solvent_batch = Batch.from_data_list(list(solvents))
    return solute_batch, solvent_batch, torch.stack(temps), torch.stack(labels)


# -----------------------------
# Evaluation Function with Uncertainty
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    y_true_list, y_pred_list, uncertainty_list = [], [], []

    with torch.no_grad():
        for solute, solvent, temp, y in tqdm(loader):
            solute, solvent = solute.to(device), solvent.to(device)
            temp, y = temp.to(device), y.to(device)

            mu, sigma, pi = model(solute, solvent, temp)

            # Expected mean and variance from DMN
            y_hat = (pi * mu).sum(dim=-1)
            var = (pi * (sigma ** 2 + mu ** 2)).sum(dim=-1) - y_hat ** 2

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())
            uncertainty_list.append(var.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list) #; y_pred = y_pred + 1
    uncertainty = np.concatenate(uncertainty_list)

    spearman = spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Spearman: {spearman:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"Mean Uncertainty: {uncertainty.mean():.4f}, Std Uncertainty: {uncertainty.std():.4f}")

    return {"Spearman": spearman, "R2": r2, "RMSE": rmse, "MAE": mae, "Uncertainty": uncertainty, "y_true": y_true, "y_pred": y_pred}


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    with open(f'{data_dir}/water_solubility.pkl', 'rb') as f:
        data = pickle.load(f)
    train = data['train']
    test = data['val']
    # with open(f'{data_dir}/water_solubility.pkl', 'rb') as f: data = pickle.load(f)
    solprop_test = data['test']

    train_loader = DataLoader(SolubilityDataset(train), batch_size=bs, shuffle=True, collate_fn=solubility_collate_fn)
    test_loader = DataLoader(SolubilityDataset(test), batch_size=bs, shuffle=False, collate_fn=solubility_collate_fn)
    solprop_test_loader = DataLoader(SolubilityDataset(solprop_test), batch_size=bs, shuffle=False, collate_fn=solubility_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolubilityPredictor(atom_feat_dim=12, bond_feat_dim=6, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_score = -999

    for ep in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for solute, solvent, temp, y in tqdm(train_loader, desc=f"Epoch {ep}/{num_epochs}"):
            solute, solvent = solute.to(device), solvent.to(device)
            temp, y = temp.to(device), y.to(device)

            optimizer.zero_grad()
            mu, sigma, pi = model(solute, solvent, temp)
            loss = dmn_loss(y, mu, sigma, pi)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {ep} | Train Loss: {total_loss/len(train_loader):.4f}")

        val_metrics = evaluate(model, test_loader, device)
        if val_metrics["Spearman"] > best_score:
            best_score = val_metrics["Spearman"]
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Saved best model (Spearman: {best_score:.4f})")

    print("\nEvaluating best model...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    results = evaluate(model, solprop_test_loader, device)
    print(results["Spearman"], results["R2"], results["RMSE"], results["MAE"])

import argparse
import json
import os
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

from helpers import set_seed, solute_level_split


# ------------------------
# Config
# ------------------------
num_epochs = 10
bs = 32
lr = 1e-4
data_dir = './data_files'
eval_data_dir = './data_files'
ckpt_dir = './weights'
results_dir = './results'

# Fraction of the pooled (train+val+test) BigSol data assigned to each fold
# when re-splitting per seed. Matches the ~90/10 train/val ratio of the
# original static split (no held-out test fold is used here since the
# reported configuration evaluates generalization on the external
# SolProp/Leeds benchmarks instead).
SPLIT_FRACS = {'train': 0.9, 'val': 0.1}


def log10_transform(y, eps=1e-12):
    return np.log10(y + eps)


# ----------------------------------------
# Temperature Modulator
# ----------------------------------------
class TempModulator(nn.Module):
    def __init__(self, emb_dim, temp_stats_path="./data_files/bigsol_temperatureK.pkl", n_rbf=10, gamma=None):
        super().__init__()
        with open(temp_stats_path, "rb") as f:
            temps = pickle.load(f)
        if isinstance(temps, dict):
            temps = np.array(list(temps.values()))
        temps = np.array(temps, dtype=np.float32)

        t_mean, t_std = np.mean(temps), np.std(temps)
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float))
        self.register_buffer("t_std", torch.tensor(t_std, dtype=torch.float))

        z_min, z_max = -3.0, 3.0
        centers = np.linspace(z_min, z_max, n_rbf)
        self.register_buffer("rbf_centers", torch.tensor(centers, dtype=torch.float))

        if gamma is None:
            gamma = 1.0 / (2 * ((z_max - z_min) / (n_rbf - 1)) ** 2)
        self.gamma = gamma

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


class ImprovedTempModulator(nn.Module):
    """Redesigned temperature conditioning (the adopted "film_v2" module),
    replacing TempModulator above. TempModulator's RBF centers only span
    z in [-3, 3], but BigSol's actual temperature range reaches z ~ 7.7
    (and ~ -3.8 on the low end); past |z| ~ 4 every Gaussian bump decays
    to ~0 simultaneously, so the encoding collapses to a near-identical
    vector regardless of how far out the true temperature is -- verified
    empirically: predictions flatline to a constant above ~365 K.

    Three changes:
      1. RBF centers are placed at empirical quantiles of BigSol's own
         z-scored temperature distribution (not a symmetric-normal
         assumption), so resolution matches where data actually is.
      2. A smooth, monotonic overflow/underflow feature (log1p(relu(.)))
         is appended so temperatures beyond the dense RBF coverage still
         produce a distinguishable, rank-preserving signal instead of a
         flat zero.
      3. True FiLM (scale *and* shift) replaces the old sigmoid-bounded
         multiplicative-only gate, for more expressive conditioning.

    (kept in sync with ablation_study.ImprovedTempModulator, which trains
    the "film_v2" ablation variant against this same design)
    """
    def __init__(self, emb_dim, temp_stats_path="./data_files/bigsol_temperatureK.pkl", n_rbf=10):
        super().__init__()
        with open(temp_stats_path, "rb") as f:
            temps = pickle.load(f)
        if isinstance(temps, dict):
            temps = np.array(list(temps.values()))
        temps = np.array(temps, dtype=np.float32)

        t_mean, t_std = np.mean(temps), np.std(temps)
        self.register_buffer("t_mean", torch.tensor(t_mean, dtype=torch.float))
        self.register_buffer("t_std", torch.tensor(t_std, dtype=torch.float))

        z_scores = (temps - t_mean) / t_std
        quantiles = np.linspace(0.05, 0.95, n_rbf)
        centers = np.quantile(z_scores, quantiles).astype(np.float32)
        self.register_buffer("rbf_centers", torch.tensor(centers, dtype=torch.float))

        spacing = float(np.median(np.diff(np.sort(centers)))) + 1e-8
        self.gamma = 1.0 / (2 * spacing ** 2)

        # Overflow/underflow trigger at the *last RBF center*, not the
        # training-set's absolute min/max -- triggering at the absolute
        # max would mean it never fires for any in-training-range z
        # (since none can exceed the max by definition), leaving exactly
        # the "sparse tail past the RBF's dense coverage" region dead.
        self.overflow_boundary = float(centers.max())
        self.underflow_boundary = float(centers.min())

        self.fc = nn.Sequential(
            nn.Linear(n_rbf + 2, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim * 2),  # -> gamma_raw, beta
        )

    def rbf_expand(self, t_norm):
        diff = t_norm.unsqueeze(-1) - self.rbf_centers
        return torch.exp(-self.gamma * diff.pow(2))

    def forward(self, embedding, temperature):
        t_norm = (temperature - self.t_mean) / (self.t_std + 1e-8)
        t_rbf = self.rbf_expand(t_norm)
        # log1p(relu(.)) grows slowly but never saturates, so it keeps
        # differentiating (and preserving rank order) arbitrarily far
        # into the tail -- unlike tanh, which saturates within a couple
        # of units and would recreate the same collapse problem.
        overflow = torch.log1p(F.relu(t_norm - self.overflow_boundary))
        underflow = torch.log1p(F.relu(self.underflow_boundary - t_norm))
        feat = torch.cat([t_rbf, overflow.unsqueeze(-1), underflow.unsqueeze(-1)], dim=-1)
        film = self.fc(feat.to(embedding.device))
        gamma_raw, beta = film.chunk(2, dim=-1)
        gamma = 1.0 + torch.tanh(gamma_raw)  # centered at 1, bounded (0, 2)
        return embedding * gamma + beta


# -----------------------------
# Deep Mixture Network (DMN)
# -----------------------------
class DMNHead(nn.Module):
    def __init__(self, input_dim, n_components=3):
        super().__init__()
        self.n_components = n_components
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
        )
        self.mu = nn.Linear(input_dim // 4, n_components)
        self.log_sigma = nn.Linear(input_dim // 4, n_components)
        self.pi_logits = nn.Linear(input_dim // 4, n_components)

    def forward(self, x):
        h = self.fc(x)
        mu = self.mu(h)
        sigma = torch.exp(self.log_sigma(h))
        pi = F.softmax(self.pi_logits(h), dim=-1)
        return mu, sigma, pi


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
class EncoderTrunk(nn.Module):
    """Solute/solvent embedding + cross-attention + AttentiveFP encoders +
    ImprovedTempModulator (film_v2) temperature conditioning. Submodule
    names (solute_embed, cross_attn_solute, ..., temp_mod_solvent) match
    ablation_study.EncoderTrunk's film_v2 configuration exactly, so a
    checkpoint saved from SolubilityPredictor below loads directly into
    AblationModel(**VARIANT_CONFIG['film_v2']) (what inference.py uses)
    and vice versa -- both are the same architecture."""
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.solute_embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.solvent_embed = nn.Linear(atom_feat_dim, hidden_dim)

        self.cross_attn_solute = CrossAttentionLayer(hidden_dim, num_heads=num_heads)
        self.cross_attn_solvent = CrossAttentionLayer(hidden_dim, num_heads=num_heads)

        self.solute_encoder = AttentiveFP(
            in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=bond_feat_dim, num_layers=2, num_timesteps=2)
        self.solvent_encoder = AttentiveFP(
            in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim,
            edge_dim=bond_feat_dim, num_layers=1, num_timesteps=1)

        self.temp_mod_solvent = ImprovedTempModulator(hidden_dim)
        self.out_dim = hidden_dim * 2

    def _per_sample_cross_attention(self, solute_x, solute_batch, solvent_x, solvent_batch):
        num_graphs = int(solute_batch.max().item()) + 1
        solute_out = torch.zeros_like(solute_x)
        solvent_out = torch.zeros_like(solvent_x)

        for i in range(num_graphs):
            s_mask, v_mask = (solute_batch == i), (solvent_batch == i)
            if s_mask.sum() == 0 or v_mask.sum() == 0:
                continue
            s_idx = torch.nonzero(s_mask).squeeze(-1)
            v_idx = torch.nonzero(v_mask).squeeze(-1)
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
        return torch.cat([h_solute, h_solvent_mod], dim=-1)


class SolubilityPredictor(nn.Module):
    """The adopted EnSol model: EncoderTrunk (cross-attention + AttentiveFP
    + redesigned FiLM temperature conditioning) + DMN head."""
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=128, n_components=3, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trunk = EncoderTrunk(atom_feat_dim, bond_feat_dim, hidden_dim, num_heads)
        self.head = DMNHead(input_dim=self.trunk.out_dim, n_components=n_components)

    def forward(self, solute_data, solvent_data, temperature):
        h = self.trunk(solute_data, solvent_data, temperature)
        mu, sigma, pi = self.head(h)
        return mu, sigma, pi


# -----------------------------
# Dataset + Collate
# -----------------------------
class SolubilityDataset(Dataset):
    def __init__(self, data_list, apply_transform=True):
        self.data_list = data_list
        self.apply_transform = apply_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        solute, solvent, temp = item[0], item[1], item[2]
        # index 3 is logS for both the 4-tuple (BigSol/SolProp: solute,
        # solvent, temp, logS) and 5-tuple (Leeds: ..., logS, feature_list)
        # formats -- item[-1] is wrong for Leeds, where it grabs the
        # trailing 41-dim feature vector instead of the label.
        label = item[3]
        if self.apply_transform:
            label = log10_transform(label)
        return solute, solvent, torch.tensor(temp, dtype=torch.float), torch.tensor(label, dtype=torch.float)


def solubility_collate_fn(batch):
    solutes, solvents, temps, labels = zip(*batch)
    return (
        Batch.from_data_list(list(solutes)),
        Batch.from_data_list(list(solvents)),
        torch.stack(temps),
        torch.stack(labels),
    )


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, device):
    model.eval()
    y_true_list, y_pred_list, uncertainty_list = [], [], []

    with torch.no_grad():
        for solute, solvent, temp, y in tqdm(loader):
            solute, solvent = solute.to(device), solvent.to(device)
            temp, y = temp.to(device), y.to(device)

            mu, sigma, pi = model(solute, solvent, temp)

            y_hat = (pi * mu).sum(dim=-1)
            var = (pi * (sigma ** 2 + mu ** 2)).sum(dim=-1) - y_hat ** 2

            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())
            uncertainty_list.append(var.cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    uncertainty = np.concatenate(uncertainty_list)

    spearman = spearmanr(y_true, y_pred)[0]
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Spearman: {spearman:.4f}, R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"Mean Uncertainty: {uncertainty.mean():.4f}, Std Uncertainty: {uncertainty.std():.4f}")

    return {"Spearman": spearman, "R2": r2, "RMSE": rmse, "MAE": mae, "Uncertainty": uncertainty}


def load_eval_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return sum(data.values(), [])
    return data


def celsius_to_kelvin_items(items):
    """leeds_training_data.pkl stores temperature in Celsius (values
    cluster at 20.0/25.0 -- standard lab room temps), unlike
    bigsol/solprop which are already in Kelvin. The model's temperature
    module is normalized against BigSol's Kelvin stats, so Leeds
    temperatures must be converted before evaluation -- see TODO.md §1.
    """
    return [[item[0], item[1], item[2] + 273.15] + list(item[3:]) for item in items]


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                         help='random seed; also reseeds the solute-level train/val re-split')
    parser.add_argument('--split', type=int, default=1,
                         help='index of the bigsol_split{N}_training.pkl file to load')
    args = parser.parse_args()

    set_seed(args.seed)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    best_model_path = f"{ckpt_dir}/bigsol_cross_attention{args.split}_seed{args.seed}.pt"

    with open(f'{data_dir}/bigsol_split{args.split}_training.pkl', 'rb') as f:
        data = pickle.load(f)
    all_items = data['train'] + data['val'] + data['test']
    resplit = solute_level_split(all_items, seed=args.seed, fracs=SPLIT_FRACS)
    train = resplit['train']
    val = resplit['val']

    # bigsol_split{N}_training.pkl now stores experimental_logS [mol/L]
    # directly (see rebuild_bigsol_pkl.py) -- already log10(mol/L), same
    # convention as SolProp/Leeds below, so no further log10_transform.
    train_loader = DataLoader(SolubilityDataset(train, apply_transform=False), batch_size=bs, shuffle=True, collate_fn=solubility_collate_fn)
    val_loader = DataLoader(SolubilityDataset(val, apply_transform=False), batch_size=bs, shuffle=False, collate_fn=solubility_collate_fn)

    solprop_data = load_eval_pkl(f'{eval_data_dir}/solprop_split1_training.pkl')
    leeds_data = celsius_to_kelvin_items(load_eval_pkl(f'{eval_data_dir}/leeds_training_data.pkl'))
    solprop_loader = DataLoader(SolubilityDataset(solprop_data, apply_transform=False), batch_size=bs, shuffle=False, collate_fn=solubility_collate_fn)
    leeds_loader = DataLoader(SolubilityDataset(leeds_data, apply_transform=False), batch_size=bs, shuffle=False, collate_fn=solubility_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolubilityPredictor(atom_feat_dim=12, bond_feat_dim=6, hidden_dim=256).to(device)
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

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["Spearman"] > best_score:
            best_score = val_metrics["Spearman"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (Spearman: {best_score:.4f})")

    print("\n--- Final Evaluation (best model) ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("SolProp:")
    solprop_metrics = evaluate(model, solprop_loader, device)

    print("Leeds:")
    leeds_metrics = evaluate(model, leeds_loader, device)

    def scalar_metrics(m):
        return {k: float(v) if np.isscalar(v) else float(np.mean(v)) for k, v in m.items()}

    results = {
        'seed': args.seed,
        'split': args.split,
        'best_val_spearman': float(best_score),
        'solprop': scalar_metrics(solprop_metrics),
        'leeds': scalar_metrics(leeds_metrics),
    }
    results_path = f"{results_dir}/bigsol_split{args.split}_seed{args.seed}_metrics.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {results_path}")

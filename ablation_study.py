"""
Ablation study: three variants of EnSol, each isolating one design choice,
trained/evaluated identically to the main model (train.py) for direct
comparability -- same BigSol data, same solute-level split, same 5 seeds
(42, 0, 1, 2, 3), same 90:10 train:val, same SolProp/Leeds test benchmarks
(Leeds temperature Celsius->Kelvin corrected).

Variants (selected via --variant):
  mse_loss      -- DMN head replaced by a plain scalar-regression head,
                    trained with MSE instead of mixture NLL.
  dmn_k5        -- same DMN head, but K=5 mixture components instead of 3.
  film_concat   -- FiLM/RBF temperature conditioning replaced by a
                    normalized temperature scalar concatenated onto the
                    solute+solvent embedding before the head.
  film_v2       -- the adopted EnSol temperature-conditioning module
                    (quantile RBF centers + overflow/underflow feature +
                    true FiLM); see TODO.md "Temperature-conditioning
                    module".
  no_cross_attn -- solute/solvent cross-attention removed; solute and
                    solvent are embedded and AttentiveFP-encoded
                    independently, then concatenated (with film_v2
                    temperature conditioning on the solvent embedding)
                    exactly as in EnSol otherwise.
  mpnn          -- EnSol (film_v2) with AttentiveFP swapped for a
                    classic MPNN (Gilmer et al. 2017) graph encoder --
                    NNConv edge-conditioned message passing + GRU node
                    update + Set2Set readout -- everything else
                    (cross-attention, film_v2 temp conditioning, DMN
                    head) unchanged.
  one_hot_solvent -- solvent graph encoder replaced by a plain lookup
                    embedding keyed by solvent identity (SMILES string),
                    no molecular structure at all; solvents outside the
                    training vocabulary (BigSol has only 70 unique
                    solvents) collapse to a learned UNK vector. No
                    cross-attention (there are no solvent node features
                    to attend over). Solute side (AttentiveFP) and
                    film_v2 temperature conditioning unchanged.

All variants share one EncoderTrunk (solute/solvent embedding,
optional cross-attention, AttentiveFP encoders) so only the head, the
temperature-conditioning method, or the cross-attention step differs
between variants and the main model -- not separately duplicated
architectures.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from helpers import set_seed, solute_level_split
from train import (CrossAttentionLayer, TempModulator, DMNHead, dmn_loss,
                    SolubilityDataset, solubility_collate_fn, load_eval_pkl,
                    celsius_to_kelvin_items, data_dir, eval_data_dir, SPLIT_FRACS)

RESULTS_DIR = './results'
CKPT_DIR = './weights'
SEEDS = [42, 0, 1, 2, 3]
VARIANTS = ['mse_loss', 'dmn_k5', 'film_concat', 'film_v2', 'no_cross_attn', 'mpnn', 'one_hot_solvent']


# -----------------------------
# Shared trunk: embedding + cross-attention + AttentiveFP encoders (+ temperature)
# -----------------------------
class TempNorm(nn.Module):
    """Just the z-score normalization piece of TempModulator, with no
    RBF expansion or gating -- for the film_concat variant."""
    def __init__(self, temp_stats_path="./data_files/bigsol_temperatureK.pkl"):
        super().__init__()
        with open(temp_stats_path, "rb") as f:
            temps = pickle.load(f)
        if isinstance(temps, dict):
            temps = np.array(list(temps.values()))
        temps = np.array(temps, dtype=np.float32)
        self.register_buffer("t_mean", torch.tensor(np.mean(temps), dtype=torch.float))
        self.register_buffer("t_std", torch.tensor(np.std(temps), dtype=torch.float))

    def forward(self, temperature):
        return (temperature - self.t_mean) / (self.t_std + 1e-8)


class ImprovedTempModulator(nn.Module):
    """Redesigned temperature conditioning, targeting a diagnosed failure
    mode of TempModulator: its RBF centers only span z in [-3, 3], but
    BigSol's actual temperature range reaches z ~ 7.7 (and ~ -3.8 on the
    low end). Past |z| ~ 4, every Gaussian bump decays to ~0 simultaneously,
    so the encoding collapses to a near-identical vector regardless of how
    far out the true temperature is -- verified empirically: predictions
    flatline to a constant above ~365 K (see TODO.md "Temperature-
    conditioning probing").

    Two changes:
      1. RBF centers are placed at empirical quantiles of BigSol's own
         z-scored temperature distribution (not a symmetric-normal
         assumption), so resolution matches where data actually is.
      2. A smooth, monotonic overflow/underflow feature (tanh(relu(.)))
         is appended so temperatures beyond the dense RBF coverage still
         produce a distinguishable, rank-preserving signal instead of a
         flat zero.
      3. True FiLM (scale *and* shift) replaces the old sigmoid-bounded
         multiplicative-only gate, for more expressive conditioning.
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


class RegressionHead(nn.Module):
    """Plain scalar regression head for the mse_loss variant -- same
    trunk-to-hidden shape as DMNHead, but a single output, no mixture."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
        )

    def forward(self, x):
        return self.fc(x).squeeze(-1)


class MPNNEncoder(nn.Module):
    """Classic MPNN (Gilmer et al. 2017): an edge-conditioned NNConv message
    function + GRU node-state update, repeated `num_steps` times, followed
    by a Set2Set graph-level readout -- used in place of AttentiveFP for
    the `mpnn` ablation variant. Same in/out interface as AttentiveFP
    (node features in, one out_channels-dim vector per graph out), so it
    drops into EncoderTrunk without touching anything else."""
    def __init__(self, hidden_dim, edge_dim, num_steps=3, out_channels=None):
        super().__init__()
        out_channels = out_channels or hidden_dim
        self.num_steps = num_steps
        edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )
        self.conv = NNConv(hidden_dim, hidden_dim, edge_net, aggr='mean')
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.out_proj = nn.Linear(hidden_dim * 2, out_channels)  # Set2Set doubles the dim

    def forward(self, x, edge_index, edge_attr, batch):
        h = x.unsqueeze(0)  # GRU hidden state: (1, N, hidden_dim)
        out = x
        for _ in range(self.num_steps):
            m = F.relu(self.conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        graph_emb = self.set2set(out, batch)
        return self.out_proj(graph_emb)


class SolventOneHotEncoder(nn.Module):
    """Replaces the solvent graph encoder entirely with a plain lookup
    embedding keyed by solvent identity (SMILES string) -- no molecular
    structure at all, just "which of the N training solvents is this."
    Solvents outside the training vocabulary (e.g. SolProp/Leeds solvents
    never seen in BigSol) collapse to a single learned UNK vector, since a
    one-hot lookup has no way to generalize to an unseen identity."""
    def __init__(self, vocab, hidden_dim):
        super().__init__()
        self.vocab = {s: i for i, s in enumerate(vocab)}
        self.unk_idx = len(vocab)
        self.embedding = nn.Embedding(len(vocab) + 1, hidden_dim)

    def forward(self, smiles_list, device):
        idx = torch.tensor([self.vocab.get(s, self.unk_idx) for s in smiles_list],
                            dtype=torch.long, device=device)
        return self.embedding(idx)


class EncoderTrunk(nn.Module):
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=256, num_heads=4, temp_mode='film',
                 use_cross_attention=True, encoder_type='attentive_fp',
                 solvent_encoding='graph', solvent_vocab=None):
        super().__init__()
        assert temp_mode in ('film', 'concat', 'film_v2')
        assert encoder_type in ('attentive_fp', 'mpnn')
        assert solvent_encoding in ('graph', 'one_hot')
        if solvent_encoding == 'one_hot':
            # no per-atom solvent features exist to cross-attend over
            assert not use_cross_attention, 'one_hot solvent_encoding requires use_cross_attention=False'
            assert solvent_vocab is not None
        self.temp_mode = temp_mode
        self.use_cross_attention = use_cross_attention
        self.solvent_encoding = solvent_encoding

        self.solute_embed = nn.Linear(atom_feat_dim, hidden_dim)
        if use_cross_attention:
            self.cross_attn_solute = CrossAttentionLayer(hidden_dim, num_heads=num_heads)
            self.cross_attn_solvent = CrossAttentionLayer(hidden_dim, num_heads=num_heads)

        if encoder_type == 'attentive_fp':
            # depth mirrors the main model: solute gets 2 layers/timesteps,
            # solvent gets 1 -- kept the same asymmetry for the mpnn variant
            # (solute num_steps=2, solvent num_steps=1) for comparability.
            self.solute_encoder = AttentiveFP(
                in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim,
                edge_dim=bond_feat_dim, num_layers=2, num_timesteps=2)
        else:
            self.solute_encoder = MPNNEncoder(hidden_dim, edge_dim=bond_feat_dim, num_steps=2, out_channels=hidden_dim)

        if solvent_encoding == 'one_hot':
            self.solvent_lookup = SolventOneHotEncoder(solvent_vocab, hidden_dim)
        else:
            self.solvent_embed = nn.Linear(atom_feat_dim, hidden_dim)
            if encoder_type == 'attentive_fp':
                self.solvent_encoder = AttentiveFP(
                    in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim,
                    edge_dim=bond_feat_dim, num_layers=1, num_timesteps=1)
            else:
                self.solvent_encoder = MPNNEncoder(hidden_dim, edge_dim=bond_feat_dim, num_steps=1, out_channels=hidden_dim)

        if temp_mode == 'film':
            self.temp_mod_solvent = TempModulator(hidden_dim)
            self.out_dim = hidden_dim * 2
        elif temp_mode == 'film_v2':
            self.temp_mod_solvent = ImprovedTempModulator(hidden_dim)
            self.out_dim = hidden_dim * 2
        else:
            self.temp_norm = TempNorm()
            self.out_dim = hidden_dim * 2 + 1

    def _per_sample_cross_attention(self, solute_x, solute_batch, solvent_x, solvent_batch):
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

        if self.solvent_encoding == 'one_hot':
            h_solute = self.solute_encoder(solute_x, solute_data.edge_index, solute_data.edge_attr, batch=solute_data.batch)
            h_solvent = self.solvent_lookup(solvent_data.smiles, device=h_solute.device)
        else:
            solvent_x = self.solvent_embed(solvent_data.x)
            if self.use_cross_attention:
                solute_x, solvent_x = self._per_sample_cross_attention(
                    solute_x, solute_data.batch, solvent_x, solvent_data.batch)
            h_solute = self.solute_encoder(solute_x, solute_data.edge_index, solute_data.edge_attr, batch=solute_data.batch)
            h_solvent = self.solvent_encoder(solvent_x, solvent_data.edge_index, solvent_data.edge_attr, batch=solvent_data.batch)

        if self.temp_mode in ('film', 'film_v2'):
            h_solvent = self.temp_mod_solvent(h_solvent, temperature)
            return torch.cat([h_solute, h_solvent], dim=-1)
        else:
            t_norm = self.temp_norm(temperature).unsqueeze(-1)
            return torch.cat([h_solute, h_solvent, t_norm], dim=-1)


class AblationModel(nn.Module):
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=256, num_heads=4,
                 temp_mode='film', head_mode='dmn', n_components=3, use_cross_attention=True,
                 encoder_type='attentive_fp', solvent_encoding='graph', solvent_vocab=None):
        super().__init__()
        assert head_mode in ('dmn', 'mse')
        self.head_mode = head_mode
        self.trunk = EncoderTrunk(atom_feat_dim, bond_feat_dim, hidden_dim, num_heads, temp_mode,
                                   use_cross_attention=use_cross_attention, encoder_type=encoder_type,
                                   solvent_encoding=solvent_encoding, solvent_vocab=solvent_vocab)
        if head_mode == 'dmn':
            self.head = DMNHead(input_dim=self.trunk.out_dim, n_components=n_components)
        else:
            self.head = RegressionHead(self.trunk.out_dim)

    def forward(self, solute_data, solvent_data, temperature):
        h = self.trunk(solute_data, solvent_data, temperature)
        return self.head(h)


VARIANT_CONFIG = {
    'mse_loss':      dict(head_mode='mse', temp_mode='film', n_components=3),
    'dmn_k5':        dict(head_mode='dmn', temp_mode='film', n_components=5),
    'film_concat':   dict(head_mode='dmn', temp_mode='concat', n_components=3),
    'film_v2':       dict(head_mode='dmn', temp_mode='film_v2', n_components=3),
    'no_cross_attn': dict(head_mode='dmn', temp_mode='film_v2', n_components=3, use_cross_attention=False),
    'mpnn':          dict(head_mode='dmn', temp_mode='film_v2', n_components=3, encoder_type='mpnn'),
    'one_hot_solvent': dict(head_mode='dmn', temp_mode='film_v2', n_components=3, use_cross_attention=False,
                             solvent_encoding='one_hot'),
}


def compute_loss(model, y, mu_or_pred, sigma=None, pi=None):
    if model.head_mode == 'dmn':
        return dmn_loss(y, mu_or_pred, sigma, pi)
    return F.mse_loss(mu_or_pred, y)


def forward_and_predict(model, solute, solvent, temp):
    """Returns y_hat (point prediction) regardless of head type."""
    out = model(solute, solvent, temp)
    if model.head_mode == 'dmn':
        mu, sigma, pi = out
        y_hat = (pi * mu).sum(dim=-1)
        return out, y_hat
    else:
        return out, out


def evaluate(model, loader, device):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for solute, solvent, temp, y in tqdm(loader):
            solute, solvent, temp, y = solute.to(device), solvent.to(device), temp.to(device), y.to(device)
            _, y_hat = forward_and_predict(model, solute, solvent, temp)
            y_true_list.append(y.cpu().numpy())
            y_pred_list.append(y_hat.cpu().numpy())
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return {
        'n': len(y_true),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'MSE': float(mean_squared_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'Spearman': float(spearmanr(y_true, y_pred)[0]),
        'R2': float(r2_score(y_true, y_pred)),
    }


def run_one_seed(variant, seed, args, device):
    set_seed(seed)
    cfg = VARIANT_CONFIG[variant]

    with open(f'{data_dir}/bigsol_split1_training.pkl', 'rb') as f:
        data = pickle.load(f)
    all_items = data['train'] + data['val'] + data['test']
    split = solute_level_split(all_items, seed=seed, fracs=SPLIT_FRACS)
    train_items, val_items = split['train'], split['val']

    train_loader = DataLoader(SolubilityDataset(train_items, apply_transform=False), batch_size=args.batch_size,
                               shuffle=True, collate_fn=solubility_collate_fn)
    val_loader = DataLoader(SolubilityDataset(val_items, apply_transform=False), batch_size=args.batch_size,
                             shuffle=False, collate_fn=solubility_collate_fn)

    solprop_data = load_eval_pkl(f'{eval_data_dir}/solprop_split1_training.pkl')
    leeds_data = celsius_to_kelvin_items(load_eval_pkl(f'{eval_data_dir}/leeds_training_data.pkl'))
    solprop_loader = DataLoader(SolubilityDataset(solprop_data, apply_transform=False), batch_size=args.batch_size,
                                 shuffle=False, collate_fn=solubility_collate_fn)
    leeds_loader = DataLoader(SolubilityDataset(leeds_data, apply_transform=False), batch_size=args.batch_size,
                               shuffle=False, collate_fn=solubility_collate_fn)

    if cfg.get('solvent_encoding') == 'one_hot':
        # vocab built from this seed's *training* items only (not val/test)
        # -- solute-level split means solvents are shared across splits
        # anyway, so this covers essentially all of them; any SolProp/Leeds
        # solvent never seen in BigSol training falls back to UNK.
        solvent_vocab = sorted(set(it[1].smiles for it in train_items))
        print(f'[{variant} seed {seed}] solvent vocab size: {len(solvent_vocab)}')
        cfg = {**cfg, 'solvent_vocab': solvent_vocab}

    model = AblationModel(atom_feat_dim=12, bond_feat_dim=6, hidden_dim=256, **cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = f'{CKPT_DIR}/ablation_{variant}_seed{seed}.pt'
    best_val_spearman = -999

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for solute, solvent, temp, y in tqdm(train_loader, desc=f'[{variant} seed {seed}] epoch {ep}/{args.epochs}'):
            solute, solvent, temp, y = solute.to(device), solvent.to(device), temp.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(solute, solvent, temp)
            if model.head_mode == 'dmn':
                mu, sigma, pi = out
                loss = compute_loss(model, y, mu, sigma, pi)
            else:
                loss = compute_loss(model, y, out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"[{variant} seed {seed}] epoch {ep}: train_loss={total_loss/len(train_loader):.4f} "
              f"val_Spearman={val_metrics['Spearman']:.4f}")
        if val_metrics['Spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['Spearman']
            torch.save(model.state_dict(), ckpt_path)
            print(f'[{variant} seed {seed}] saved new best (val Spearman={best_val_spearman:.4f})')

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f'[{variant} seed {seed}] SolProp:')
    solprop_metrics = evaluate(model, solprop_loader, device)
    print(f'[{variant} seed {seed}] Leeds:')
    leeds_metrics = evaluate(model, leeds_loader, device)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {'variant': variant, 'seed': seed, 'best_val_spearman': float(best_val_spearman),
              'solprop': solprop_metrics, 'leeds': leeds_metrics}
    with open(f'{RESULTS_DIR}/ablation_{variant}_seed{seed}_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=VARIANTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Variant: {args.variant} ({VARIANT_CONFIG[args.variant]})')

    results = [run_one_seed(args.variant, seed, args, device) for seed in args.seeds]

    def agg(bench, key):
        vals = [r[bench][key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {}
    for bench in ['solprop', 'leeds']:
        print(f'\n=== {args.variant} {bench} summary across seeds ===')
        summary[bench] = {}
        for key in ['MAE', 'MSE', 'RMSE', 'Spearman', 'R2']:
            m, s = agg(bench, key)
            summary[bench][key] = {'mean': m, 'std': s}
            print(f'{key}: {m:.4f} +/- {s:.4f}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/ablation_{args.variant}_summary.json', 'w') as f:
        json.dump({'variant': args.variant, 'config': VARIANT_CONFIG[args.variant],
                    'per_seed': results, 'aggregate': summary}, f, indent=2)
    print(f'Saved summary to {RESULTS_DIR}/ablation_{args.variant}_summary.json')


if __name__ == '__main__':
    main()

"""
Transfer learning: fine-tune the BigSol-pretrained model on AqSolDB
(water solubility), to demonstrate the pretraining actually transfers.

AqSolDB has no per-sample temperature, so this uses a no-temperature
variant of the architecture (SolubilityPredictorNoTemp below) -- same
solute/solvent embedding, cross-attention, and encoder stack as
model.SolubilityPredictor, minus TempModulator. Pretrained weights are
loaded key-by-key so only the matching (non-temperature) layers transfer.

5 seeds, each with its own 8:1:1 scaffold split (Murcko scaffolds,
shuffled by seed before greedy bucket assignment -- see
helpers.scaffold_split): train on train, checkpoint on best val
Spearman, report MAE/MSE/RMSE/Spearman/R2 on test, then mean +/- std
across seeds.

Run `prepare_aqsoldb.py` first to build data_files/aqsoldb_water_finetune.pkl.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from helpers import set_seed, scaffold_split
from model import CrossAttentionLayer, DMNHead, dmn_loss

DATA_PATH = './data_files/aqsoldb_water_finetune.pkl'
RESULTS_DIR = './results'
CKPT_DIR = './weights'
SPLIT_FRACS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
SEEDS = [0, 1, 2, 3, 4]


# -----------------------------
# Model: same as model.SolubilityPredictor, minus temperature module
# -----------------------------
class SolubilityPredictorNoTemp(nn.Module):
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_dim=256, n_components=3, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

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

        # no temp_mod_solvent -- no per-sample temperature in this data
        self.dmn = DMNHead(input_dim=hidden_dim * 2, n_components=n_components)

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

    def forward(self, solute_data, solvent_data):
        solute_x = self.solute_embed(solute_data.x)
        solvent_x = self.solvent_embed(solvent_data.x)
        solute_x_cross, solvent_x_cross = self._per_sample_cross_attention(
            solute_x, solute_data.batch, solvent_x, solvent_data.batch)

        h_solute = self.solute_encoder(solute_x_cross, solute_data.edge_index, solute_data.edge_attr, batch=solute_data.batch)
        h_solvent = self.solvent_encoder(solvent_x_cross, solvent_data.edge_index, solvent_data.edge_attr, batch=solvent_data.batch)
        h = torch.cat([h_solute, h_solvent], dim=-1)
        mu, sigma, pi = self.dmn(h)
        return mu, sigma, pi


def load_pretrained_partial(model, pretrained_path, device):
    """Load only the keys that exist in both the pretrained (temperature-
    aware) checkpoint and this no-temperature model, i.e. everything
    except temp_mod_solvent.* -- which this architecture doesn't have.

    The pretrained checkpoint may come from either SolubilityPredictor
    (train.py/model.py -- unprefixed keys, head named `dmn.*`) or
    AblationModel (ablation_study.py -- trunk submodule, so keys are
    prefixed `trunk.*`, and the head is named `head.*`). Both are
    normalized to the unprefixed/`dmn.*` naming this model uses before
    matching, so pretrained transfer works regardless of which one
    produced the checkpoint.
    """
    pretrained_state = torch.load(pretrained_path, map_location=device)
    renamed_state = {}
    for k, v in pretrained_state.items():
        k = k.removeprefix('trunk.')
        if k.startswith('head.'):
            k = 'dmn.' + k[len('head.'):]
        renamed_state[k] = v
    model_state = model.state_dict()
    matched = {k: v for k, v in renamed_state.items()
               if k in model_state and model_state[k].shape == v.shape}
    n_skipped = len(renamed_state) - len(matched)
    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f'Loaded {len(matched)}/{len(model_state)} matching pretrained tensors from '
          f'{pretrained_path} ({n_skipped} skipped, e.g. the temperature module)')


# -----------------------------
# Data
# -----------------------------
class WaterSolubilityDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        solute, solvent, label = self.data_list[idx]
        return solute, solvent, torch.tensor(label, dtype=torch.float)


def collate_fn(batch):
    solutes, solvents, labels = zip(*batch)
    return (Batch.from_data_list(list(solutes)),
            Batch.from_data_list(list(solvents)),
            torch.stack(labels))


def evaluate(model, loader, device):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for solute, solvent, y in loader:
            solute, solvent, y = solute.to(device), solvent.to(device), y.to(device)
            mu, sigma, pi = model(solute, solvent)
            y_hat = (pi * mu).sum(dim=-1)
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


# -----------------------------
# One seed: split, fine-tune, test
# -----------------------------
def run_one_seed(seed, args, device):
    set_seed(seed)

    with open(args.data_path, 'rb') as f:
        items = pickle.load(f)
    split = scaffold_split(items, seed=seed, fracs=SPLIT_FRACS)
    train_items, val_items, test_items = split['train'], split['val'], split['test']
    print(f'[seed {seed}] train={len(train_items)} val={len(val_items)} test={len(test_items)}')

    train_loader = DataLoader(WaterSolubilityDataset(train_items), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(WaterSolubilityDataset(val_items), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(WaterSolubilityDataset(test_items), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    model = SolubilityPredictorNoTemp(atom_feat_dim=12, bond_feat_dim=6, hidden_dim=256).to(device)
    tag = ('randominit' if args.random_init else 'pretrained') + args.tag_suffix
    if args.random_init:
        print(f'[seed {seed}] random init (no pretrained weights loaded) -- ablation arm')
    else:
        load_pretrained_partial(model, args.pretrained_path, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = f'{CKPT_DIR}/water_solubility_transfer_{tag}_seed{seed}.pt'
    best_val_spearman = -999

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for solute, solvent, y in tqdm(train_loader, desc=f'[seed {seed}] epoch {ep}/{args.epochs}'):
            solute, solvent, y = solute.to(device), solvent.to(device), y.to(device)
            optimizer.zero_grad()
            mu, sigma, pi = model(solute, solvent)
            loss = dmn_loss(y, mu, sigma, pi)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        print(f"[seed {seed}] epoch {ep}: train_loss={total_loss/len(train_loader):.4f} "
              f"val_Spearman={val_metrics['Spearman']:.4f}")
        if val_metrics['Spearman'] > best_val_spearman:
            best_val_spearman = val_metrics['Spearman']
            torch.save(model.state_dict(), ckpt_path)
            print(f'[seed {seed}] saved new best (val Spearman={best_val_spearman:.4f})')

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print(f"[seed {seed}] TEST: MAE={test_metrics['MAE']:.4f} MSE={test_metrics['MSE']:.4f} "
          f"RMSE={test_metrics['RMSE']:.4f} Spearman={test_metrics['Spearman']:.4f} R2={test_metrics['R2']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {'seed': seed, 'init': tag, 'best_val_spearman': float(best_val_spearman), 'test': test_metrics}
    with open(f'{RESULTS_DIR}/water_solubility_transfer_{tag}_seed{seed}_metrics.json', 'w') as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str, default='./weights/bigsol_cross_attention.pt',
                         help='pretrained BigSol checkpoint to fine-tune from. Point this at a '
                              'corrected post-relabel-fix checkpoint (e.g. '
                              './weights/bigsol_cross_attention1_seed42.pt) once retraining finishes '
                              '-- see TODO.md §2 for why the pre-fix checkpoint is not final.')
    parser.add_argument('--random_init', action='store_true',
                         help='Ablation arm: skip loading pretrained weights, train from random '
                              'init instead. Same data, splits, seeds, and architecture as the '
                              'pretrained run -- only the initialization differs.')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                         help='pkl of [solute_Data, solvent_Data, logS] items to fine-tune on. '
                              'Defaults to the full ~10K AqSolDB set; point at a subsampled pkl '
                              '(e.g. aqsoldb_water_finetune_5k.pkl) to study dataset-size effects.')
    parser.add_argument('--tag_suffix', type=str, default='',
                         help='appended to checkpoint/results filenames so runs on a different '
                              'dataset (e.g. "_5k") don\'t overwrite the default-dataset run.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = ('randominit' if args.random_init else 'pretrained') + args.tag_suffix
    print(f'Using device: {device}')
    print(f'Data: {args.data_path}')
    print('Random init (ablation)' if args.random_init else f'Fine-tuning from: {args.pretrained_path}')

    results = [run_one_seed(seed, args, device) for seed in args.seeds]

    def agg(key):
        vals = [r['test'][key] for r in results]
        return float(np.mean(vals)), float(np.std(vals))

    print(f'\n=== Transfer learning summary across seeds ({tag}) ===')
    summary = {}
    for key in ['MAE', 'MSE', 'RMSE', 'Spearman', 'R2']:
        m, s = agg(key)
        summary[key] = {'mean': m, 'std': s}
        print(f'{key}: {m:.4f} +/- {s:.4f}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/water_solubility_transfer_{tag}_summary.json', 'w') as f:
        json.dump({'init': tag, 'pretrained_path': None if args.random_init else args.pretrained_path,
                    'per_seed': results, 'aggregate': summary}, f, indent=2)
    print(f'Saved summary to {RESULTS_DIR}/water_solubility_transfer_{tag}_summary.json')


if __name__ == '__main__':
    main()

import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
from helpers import smiles_to_data
from model import SolubilityDataset, solubility_collate_fn, evaluate
from ablation_study import AblationModel, VARIANT_CONFIG

# The adopted EnSol model: cross-attention + AttentiveFP encoders + the
# redesigned FiLM temperature module (quantile-based RBF centers, true
# gamma*x+beta) + DMN head -- see TODO.md's "Temperature-conditioning
# module" note. This is the checkpoint family whose numbers are reported
# for EnSol on SolProp/Leeds; `ablation_film_v2_seed*.pt` is trained
# identically, just via ablation_study.py's shared trunk.
best_model_path = './weights/ablation_film_v2_seed42.pt'


def predict(df, bs=32, verbose=True):
    if verbose:
        print('Preparing data ...')
    d = []
    for i in trange(len(df)):
        solute_smiles = df.loc[i, 'solute_smiles']
        solvent_smiles = df.loc[i, 'solvent_smiles']
        temperature = df.loc[i, 'temperature']
        experimental_logS = df.loc[i, 'experimental_logS [mol/L]']

        solute = smiles_to_data(solute_smiles)
        solute.smiles = solute_smiles
        solvent = smiles_to_data(solvent_smiles)
        solvent.smiles = solvent_smiles
        d.append([solute, solvent, temperature, experimental_logS])

    test_loader = DataLoader(SolubilityDataset(d), batch_size=bs, shuffle=False, collate_fn=solubility_collate_fn)

    if verbose:
        print('Predicting ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AblationModel(atom_feat_dim=12, bond_feat_dim=6, hidden_dim=256, **VARIANT_CONFIG['film_v2']).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    metrics = evaluate(model, test_loader, device)
    if verbose:
        print(metrics)
    return metrics

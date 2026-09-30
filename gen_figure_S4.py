"""
Builds the source data for Supplementary Figure S4: distributional
agreement between EnSol predictions and ground-truth LogS values on
SolProp (panel a) and Leeds (panel b) -- see EnSol_SI.docx caption.

Runs the adopted EnSol checkpoint (weights/bigsol_cross_attention.pt,
seed 42) on the full SolProp and Leeds sets and writes a flat
(dataset, y_true, y_pred) CSV, directly usable for an overlaid
groundtruth-vs-prediction histogram per dataset, e.g.:

    df = pd.read_csv('SI_results/figure_S4.csv')
    sub = df[df.dataset == 'SolProp']
    plt.hist(sub['y_true'], bins=60, alpha=0.6, label='Groundtruth')
    plt.hist(sub['y_pred'], bins=60, alpha=0.6, label='Prediction')
"""
import torch
import pandas as pd
from torch_geometric.data import Batch

from train import load_eval_pkl, celsius_to_kelvin_items, eval_data_dir
from ablation_study import AblationModel, VARIANT_CONFIG

OUT_DIR = './SI_results'
CKPT_PATH = './weights/bigsol_cross_attention.pt'


def predict(model, items, device, batch_size=64):
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            solute_batch = Batch.from_data_list([it[0] for it in chunk]).to(device)
            solvent_batch = Batch.from_data_list([it[1] for it in chunk]).to(device)
            temps = torch.tensor([it[2] for it in chunk], dtype=torch.float).to(device)
            mu, sigma, pi = model(solute_batch, solvent_batch, temps)
            y_hat = (pi * mu).sum(dim=-1)
            y_true.extend([float(it[3]) for it in chunk])
            y_pred.extend(y_hat.cpu().numpy().tolist())
    return y_true, y_pred


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AblationModel(atom_feat_dim=12, bond_feat_dim=6, hidden_dim=256,
                           **VARIANT_CONFIG['film_v2']).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    solprop = load_eval_pkl(f'{eval_data_dir}/solprop_split1_training.pkl')
    leeds = celsius_to_kelvin_items(load_eval_pkl(f'{eval_data_dir}/leeds_training_data.pkl'))

    st, sp = predict(model, solprop, device)
    lt, lp = predict(model, leeds, device)

    df = pd.concat([
        pd.DataFrame({'dataset': 'SolProp', 'y_true': st, 'y_pred': sp}),
        pd.DataFrame({'dataset': 'Leeds', 'y_true': lt, 'y_pred': lp}),
    ])
    df.to_csv(f'{OUT_DIR}/figure_S4.csv', index=False)
    print(f'SolProp: n={len(st)}  y_true=[{min(st):.3f}, {max(st):.3f}]  y_pred=[{min(sp):.3f}, {max(sp):.3f}]')
    print(f'Leeds:   n={len(lt)}  y_true=[{min(lt):.3f}, {max(lt):.3f}]  y_pred=[{min(lp):.3f}, {max(lp):.3f}]')


if __name__ == '__main__':
    main()

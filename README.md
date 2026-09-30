# EnSol: Environment-Aware Graph Neural Network for Probabilistic Prediction of Molecular Solubility

EnSol predicts molecular solubility (logS, mol/L) given a solute, a solvent, and a temperature. It combines:

- **AttentiveFP** encoders for solute and solvent molecular graphs
- **Cross-attention** between solute and solvent node features before graph pooling
- **Temperature Modulator** that conditions solvent embeddings via quantile-based RBF centers (fit to the training temperature distribution) plus a monotonic overflow/underflow feature, applied through a true FiLM step (`γ·x + β`)
- **Deep Mixture Network (DMN)** head that outputs a mixture of 3 Gaussians, providing both a point estimate and calibrated uncertainty

---

## Repository Structure

```
EnSol/
├── train.py                                            # Main training script (BigSol dataset)
├── model.py                                            # Shared model components (CrossAttentionLayer, DMNHead, dmn_loss,
│                                                       #   SolubilityDataset, evaluate, ...) + its own fine-tuning entrypoint
├── inference.py                                        # Inference function
├── helpers.py                                          # SMILES → PyG graph featurizer, train/val/test splitters
├── transfer_learning.py                                # Fine-tunes the adopted checkpoint on AqSolDB / ESOL
│                                                       #   (water solubility, no-temperature architecture)
├── ablation_study.py                                   # Ablation variants, incl. AblationModel/VARIANT_CONFIG
│                                                       #   (inference.py's architecture)
│
├── data_files/
│   ├── bigsol_split1_training.pkl                      # BigSol train/val/test splits
│   ├── bigsol_temperatureK.pkl                         # Temperature statistics for normalization
│   ├── solprop_split1_training.pkl                     # SolProp evaluation set
│   ├── leeds_training_data.pkl                         # Leeds evaluation set
│   ├── aqsoldb_water_finetune.pkl                      # AqSolDB fine-tune set (used by transfer_learning.py)
│   ├── BigSol.csv
│   ├── SolProp.csv
│   └── Leeds.csv
│
├── weights/                                            # Trained checkpoints (Git LFS)
│   ├── bigsol_cross_attention.pt                       # adopted EnSol model -- inference.py default
│   └── water_solubility_model.pt                       # AqSolDB fine-tune of the above (transfer_learning.py output)
│
├── SI_results/                                         # Supplementary-Information source data
│   ├── lab_testing.csv                                 # Lab-100 experimental set (100 pairs)
│   ├── figure_S1.csv / figure_S2.csv / figure_S3.csv   # BigSol/SolProp/Leeds LogS & temp distributions
│   ├── figure_S4.csv                                   # EnSol prediction vs groundtruth LogS (SolProp & Leeds)
│   ├── table_S1.csv / table_S2.csv / table_S3.csv / table_S5.csv
│   └── per_sample/                                     # per-sample predictions backing the SI tables
│       ├── Ensol_{SolProp,Leeds,lab_data}.csv
│       ├── FASTSOLV_{SolProp,Leeds,lab_data}.csv
│       └── Vermeire_{SolProp,Leeds,lab_data}.csv
│
├── data.md                                             # Public dataset citations/links used in this study
└── requirements.txt
```

---

## Dependencies

Install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version (developed with `torch==2.7.1+cu126`), then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1: Train

Training uses the BigSol dataset. Edit the config block at the top of `train.py` to set the split index, number of epochs, batch size, and output checkpoint path.

```python
# train.py
split = 2            # which BigSol split to use (1, 2, or 3)
num_epochs = 10
bs = 32
lr = 1e-4
best_model_path = f"./weights/bigsol_cross_attention{split}.pt"
data_dir = './data_files'
```

Create the checkpoint directory, then run:

```bash
python train.py
```

The script will:
1. Load `data_files/bigsol_split{split}_training.pkl` (keys: `train`, `val`, `test`)
2. Train with Adam optimizer, saving the checkpoint with the best Spearman correlation on the validation set
3. After training, evaluate the best checkpoint on **SolProp** and **Leeds** held-out sets and report Spearman, R², RMSE, and MAE

`SolubilityPredictor` uses the same architecture as `weights/bigsol_cross_attention.pt` (cross-attention + AttentiveFP + redesigned FiLM temperature module + DMN head), so a checkpoint produced here is directly interchangeable with it -- copy or rename the output to `weights/bigsol_cross_attention.pt` to use it with `inference.py` (Step 2).

---

## Step 2: Evaluate / Inference

To run inference on a custom dataset, prepare a pandas DataFrame with the following columns:

| Column | Description |
|---|---|
| `solute_smiles` | SMILES string of the solute |
| `solvent_smiles` | SMILES string of the solvent |
| `temperature` | Temperature in Kelvin |
| `experimental_logS [mol/L]` | Experimental logS (used for metrics; can be `NaN` if unknown) |

Then call:

```python
import pandas as pd
from inference import predict

df = pd.read_csv('your_data.csv')
metrics = predict(df, bs=32)
# Returns: Spearman, R2, RMSE, MAE, and per-sample uncertainty estimates
```

The `predict` function loads weights from `weights/bigsol_cross_attention.pt` by default — the adopted EnSol checkpoint (seed 42, redesigned FiLM temperature module), included in this repo via Git LFS. The model outputs a mixture-of-Gaussians prediction; the reported value is the expected mean and the uncertainty is the predictive variance across the 3 mixture components.

---

## Model Architecture Summary

| Component | Detail |
|---|---|
| Atom features | 12-dim: atomic number, electronegativity, vdW radius, formal charge, aromaticity, H count, valence, H-donor, H-acceptor, SP/SP2/SP3 hybridization |
| Bond features | 6-dim: single, double, triple, aromatic, conjugated, in-ring |
| Solute encoder | AttentiveFP (2 layers, 2 timesteps, hidden=256) |
| Solvent encoder | AttentiveFP (1 layer, 1 timestep, hidden=256) |
| Cross-attention | 4-head, applied per sample before pooling |
| Temperature encoding | 10-center quantile-based RBF on z-scored temperature + overflow/underflow feature → FiLM (`γ·x + β`) |
| Prediction head | Deep Mixture Network, 3 Gaussian components |
| Training loss | Negative log-likelihood of the Gaussian mixture |

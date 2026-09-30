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
├── train.py                      # Main training script (BigSol dataset, DMN + FiLM temp module)
├── model.py                      # Fine-tuning script (water solubility dataset)
├── inference.py                  # Inference function
├── helpers.py                    # SMILES → PyG graph featurizer, train/val/test splitters
├── ablation_study.py             # Ablation variants (MSE head, K=5, concat-temp, film_v2,
│                                  #   no-cross-attn, MPNN encoder, one-hot solvent)
├── transfer_learning.py          # Fine-tunes the BigSol-pretrained model on AqSolDB / ESOL
├── prepare_aqsoldb.py            # Builds the AqSolDB water-solubility fine-tune set
├── prepare_esol.py               # Builds the ESOL (Delaney) water-solubility fine-tune set
├── rebuild_bigsol_pkl.py         # Rebuilds bigsol_split1_training.pkl from BigSol.csv
│
├── benchmark_fastsolv.py         # FastSolv baseline on SolProp/Leeds
├── benchmark_recall.py           # Top-k solvent-recall benchmark
├── eval_lab100_ensol.py          # EnSol (old checkpoint) on the Lab-100 experimental set
├── eval_lab100_ensol_filmv2.py   # EnSol (adopted film_v2 checkpoint) on Lab-100
├── eval_lab100_fastsolv.py       # FastSolv on Lab-100
├── eval_lab100_vermeire.py       # Vermeire/SolProp_ML on Lab-100
├── ranking_eval.py               # Solute-/solvent-ranking evaluation on SolProp
├── constrained_298k_eval.py      # SolProp evaluation restricted to ~298.15 K
├── temperature_probing.py        # Per-cluster temperature-sweep probing
├── temperature_probing_film_v2.py
├── uncertainty_probing.py        # DMN mixture-variance vs. error calibration check
├── similarity_analysis.py        # Train/test structural similarity analysis
│
├── data_files/                   # Datasets (see data.md for public sources/links)
│   ├── bigsol_split1_training.pkl     # BigSol train/val/test splits
│   ├── bigsol_temperatureK.pkl        # Temperature statistics for normalization
│   ├── solprop_split1_training.pkl    # SolProp evaluation set
│   ├── leeds_training_data.pkl        # Leeds evaluation set
│   ├── aqsoldb_water_finetune*.pkl    # AqSolDB fine-tune sets (full / 5k / 1k subsamples)
│   ├── esol_water_finetune.pkl        # ESOL (Delaney) fine-tune set
│   ├── BigSol.csv / SolProp.csv / Leeds.csv / ESOL_delaney.csv / AqSolDB_curated.tab
│   └── bigsol_split1_training.MOLEFRACTION_BUG.pkl.bak  # pre-fix backup (see rebuild_bigsol_pkl.py)
│
├── weights/                      # Trained checkpoints (Git LFS)
│   ├── ablation_film_v2_seed42.pt                # adopted EnSol model -- inference.py default
│   ├── bigsol_cross_attention1_seed{0,2,42}.pt   # earlier-FiLM model, used for Table 1 Lab-100 row only
│   ├── ablation_<variant>_seed{0,1,2,3,42}.pt    # one family per other ablation variant
│   └── water_solubility_transfer_*_seed*.pt      # AqSolDB/ESOL transfer-learning checkpoints
│
├── results/                      # Raw per-run metrics JSON/logs + full per-seed prediction CSVs
├── histogram_data/                # BigSol/SolProp/Leeds solubility & temperature distributions
├── similarity_results/           # Figures + report from similarity_analysis.py
│
├── SI_results/                   # Supplementary-Information source data (see below)
│   ├── lab_testing.csv                # Lab-100 experimental set (100 pairs)
│   ├── table_S1.csv / table_S2.csv / table_S3.csv / table_S5.csv  # per-seed SI table data
│   └── per_sample/                    # per-sample predictions backing the SI tables
│       ├── Ensol_{SolProp,Leeds,lab_data}.csv
│       ├── FASTSOLV_{SolProp,Leeds,lab_data}.csv
│       └── Vermeire_{SolProp,Leeds,lab_data}.csv
│
├── data.md                       # Public dataset citations/links used in this study
├── TODO.md                       # Working notes: methodology, per-seed results, audit log
├── EnSol_SI.docx                 # Supplementary Information document
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

The `predict` function loads weights from `weights/ablation_film_v2_seed42.pt` by default — the adopted EnSol checkpoint (seed 42, redesigned FiLM temperature module), included in this repo via Git LFS. (Earlier versions of this README pointed at `weights/water_solubility_model.pt`, a checkpoint from a deprecated fine-tuning path whose source data no longer exists in this repo; `weights/bigsol_cross_attention1_seed*.pt` is a separate, earlier-FiLM checkpoint family used only for the Lab-100 "Experimental data" evaluation — see `TODO.md`.) The model outputs a mixture-of-Gaussians prediction; the reported value is the expected mean and the uncertainty is the predictive variance across the 3 mixture components.

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

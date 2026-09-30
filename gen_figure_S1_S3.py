"""
Builds the source data for Supplementary Figures S1-S3: the distribution
of experimental LogS and temperature values in BigSol, SolProp, and Leeds
(each figure's panel a = LogS histogram+KDE, panel b = temperature
histogram -- see EnSol_SI.docx captions).

Each dataset is loaded in full (train+val+test pooled, same convention as
train.py's external-benchmark evaluation and TODO.md's reported dataset
sizes: BigSol ~100,570 / SolProp 6,236 / Leeds 1,469 rows) and written to
a flat (temperature_K, logS) CSV -- directly plottable as-is (e.g.
seaborn.histplot(df['logS'], kde=True) / histplot(df['temperature_K'])).

Leeds temperatures are stored in Celsius in the raw pkl (see
train.py.celsius_to_kelvin_items) and converted to Kelvin here, matching
the docx caption ("primarily near 293 K and 298 K").
"""
import pickle
import pandas as pd

DATA_DIR = './data_files'
OUT_DIR = './SI_results'


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return sum(data.values(), []) if isinstance(data, dict) else data


def build_df(items, temp_in_celsius=False):
    temps = [float(it[2]) for it in items]
    if temp_in_celsius:
        temps = [t + 273.15 for t in temps]
    return pd.DataFrame({
        'temperature_K': temps,
        'logS': [float(it[3]) for it in items],
    })


bigsol = build_df(load_pkl(f'{DATA_DIR}/bigsol_split1_training.pkl'), temp_in_celsius=False)
bigsol.to_csv(f'{OUT_DIR}/figure_S1.csv', index=False)
print(f'figure_S1.csv (BigSol): {len(bigsol)} rows, '
      f'logS range [{bigsol.logS.min():.3f}, {bigsol.logS.max():.3f}], '
      f'temp range [{bigsol.temperature_K.min():.2f}, {bigsol.temperature_K.max():.2f}] K')

solprop = build_df(load_pkl(f'{DATA_DIR}/solprop_split1_training.pkl'), temp_in_celsius=False)
solprop.to_csv(f'{OUT_DIR}/figure_S2.csv', index=False)
print(f'figure_S2.csv (SolProp): {len(solprop)} rows, '
      f'logS range [{solprop.logS.min():.3f}, {solprop.logS.max():.3f}], '
      f'temp range [{solprop.temperature_K.min():.2f}, {solprop.temperature_K.max():.2f}] K')

# NOTE: the published Fig. S3 panel b was plotted from Leeds's raw,
# un-converted Celsius temperature values (confirmed against the docx's
# embedded image: x-axis 14-33 peaking at 20/25, i.e. degrees C, not the
# 293/298 K the caption text describes) -- kept in Celsius here to match
# the actual published figure, not the caption text.
leeds = build_df(load_pkl(f'{DATA_DIR}/leeds_training_data.pkl'), temp_in_celsius=False)
leeds = leeds.rename(columns={'temperature_K': 'temperature_C'})
leeds.to_csv(f'{OUT_DIR}/figure_S3.csv', index=False)
print(f'figure_S3.csv (Leeds): {len(leeds)} rows, '
      f'logS range [{leeds.logS.min():.3f}, {leeds.logS.max():.3f}], '
      f'temp range [{leeds.temperature_C.min():.2f}, {leeds.temperature_C.max():.2f}] degC')

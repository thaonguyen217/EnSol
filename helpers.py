from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch_geometric.data import Data

# Pauling electronegativity lookup (simplified)
ELECTRONEGATIVITY = {
    1: 2.20, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 15: 2.19,
    16: 2.58, 17: 3.16, 35: 2.96, 53: 2.66
}

# van der Waals radii (Å)
VDW_RADIUS = {
    1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98
}

def smiles_to_data(smiles: str) -> Data:
    """Convert a SMILES string to a PyTorch Geometric Data object."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    atom_features = []
    for atom in mol.GetAtoms():
        Z = atom.GetAtomicNum()
        electroneg = ELECTRONEGATIVITY.get(Z, 0.0)
        vdw = VDW_RADIUS.get(Z, 1.5)
        hyb = atom.GetHybridization()
        hyb_encoding = [
            int(hyb == rdchem.HybridizationType.SP),
            int(hyb == rdchem.HybridizationType.SP2),
            int(hyb == rdchem.HybridizationType.SP3)
        ]
        features = [
            Z,
            electroneg,
            vdw,
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.GetTotalNumHs()),
            int(atom.GetTotalValence()),
            int(atom.GetAtomicNum() in [7, 8] and atom.GetTotalNumHs() > 0),  # donor
            int(atom.GetAtomicNum() in [7, 8])  # acceptor
        ] + hyb_encoding
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edges
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bond_features = [
            1 if bond_type == rdchem.BondType.SINGLE else 0,
            1 if bond_type == rdchem.BondType.DOUBLE else 0,
            1 if bond_type == rdchem.BondType.TRIPLE else 0,
            1 if bond_type == rdchem.BondType.AROMATIC else 0,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        # undirected edges (i→j and j→i)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [bond_features, bond_features]

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    return data

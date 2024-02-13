"""
# Example Python calls:

python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_NBO_charge_Boltz.csv atom 0 trained_models/acids/C1_NBO_charge/boltz/model_best.pt 1

python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_Boltz.csv atom 0 trained_models/acids/C1_Vbur/boltz/model_best.pt 1

python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_max.csv atom 0 trained_models/acids/C1_Vbur/max/model_best.pt 1

python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_min.csv atom 0 trained_models/acids/C1_Vbur/min/model_best.pt 1

python test.py data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_Vbur_lowE.csv atom 0 trained_models/acids/C1_Vbur/min_E/model_best.pt 1

"""
import torch
import torch_geometric

import pandas as pd
import numpy as np

import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
from rdkit import Chem

from tqdm import tqdm
from copy import deepcopy
import random
import re
import os
import shutil
import argparse
import sys
import ast

from datasets.dataset_3D import *
from models.dimenetpp import *

# -------------------------------------------
# Define job parameters
# -------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("conformer_data_file", type=str) # path to conformer dataset csv file
parser.add_argument("descriptor_data_file", type=str) # path to descriptor dataset csv file
parser.add_argument("property_type", type=str) # 'bond', 'atom', 'mol'
parser.add_argument("keep_explicit_hydrogens", type = int) # 0 or 1 or 2
parser.add_argument("pretrained_path", type=str) # path to pretrained model, or str 'none'
parser.add_argument("use_atom_features", type = int) # 1

args = parser.parse_args()

property_type = args.property_type # 'bond', 'atom', or 'mol'

keep_explicit_hydrogens = bool(args.keep_explicit_hydrogens) # False or True
remove_Hs_except_functional = args.keep_explicit_hydrogens == 2 # assumes keep_explicit_hydrogens == True

pretrained_model = args.pretrained_path

num_workers = 4
use_atom_features = bool(args.use_atom_features) # default should be True

# -------------------------------------------
# Loading training data (regression targets and input conformers)

descriptor_df = pd.read_csv(args.descriptor_data_file, converters={"bond_atom_tuple": ast.literal_eval})

conformers_df = pd.read_csv(args.conformer_data_file).reset_index(drop = True)
conformers_df['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in conformers_df.mol_block]
conformers_df['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in conformers_df['mols']]

merged_df = conformers_df.merge(descriptor_df, on = 'Name_int')
test_dataframe = merged_df[merged_df.split == 'test'].reset_index(drop = True)

print(len(set(test_dataframe.Name_int)), len(test_dataframe))

# -------------------------------------------
# creating model, optimizer, dataloaders

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

if property_type == 'atom':
    atom_ID_test = np.array(list(test_dataframe.atom_index), dtype = int)
else:
    atom_ID_test = None

if property_type == 'bond':
    bond_ID_1_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 1]
else:
    bond_ID_1_test = None
    bond_ID_2_test = None

if keep_explicit_hydrogens:
    mols_test = list(test_dataframe.mols)
else:
    mols_test = list(test_dataframe.mols_noHs)

test_dataset = Dataset_3D(
    property_type = property_type,
    mols = mols_test, 
    mol_types = list(test_dataframe['mol_type']),
    targets = list(test_dataframe['y']),
    ligand_ID = np.array(test_dataframe['Name_int']),
    atom_ID = atom_ID_test,
    bond_ID_1 = bond_ID_1_test,
    bond_ID_2 = bond_ID_2_test,
    remove_Hs_except_functional = remove_Hs_except_functional,
)
test_loader = torch_geometric.loader.DataLoader(
    dataset = test_dataset,
    batch_size = 100,
    shuffle = False,
    num_workers = num_workers,
)

example_data = test_dataset[0]
atom_feature_dim = int(example_data.atom_features.shape[-1]) # 53

model = DimeNetPlusPlus(
    property_type = property_type, 
    use_atom_features = use_atom_features, 
    atom_feature_dim = atom_feature_dim if use_atom_features else 1, 
)
if pretrained_model != '':
    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)

model.to(device)

# -------------------------------------------
# testing loop definition

def loop(model, batch, property_type = 'bond'):
    
    batch = batch.to(device)
        
    if property_type == 'bond':
        out = model(
            batch.x.squeeze(), 
            batch.pos, 
            batch.batch,
            batch.atom_features,
            select_bond_start_atom_index = batch.bond_start_ID_index,
            select_bond_end_atom_index = batch.bond_end_ID_index,
        )
    
    elif property_type == 'atom':
        out = model(
            batch.x.squeeze(),
            batch.pos, 
            batch.batch,
            batch.atom_features,
            select_atom_index = batch.atom_ID_index,
        )
        
    elif property_type == 'mol':
        out = model(
            batch.x.squeeze(),
            batch.pos,
            batch.batch,
            batch.atom_features,
        )
    
    targets = batch.targets
    pred_targets = out[0].squeeze()
    mse_loss = torch.mean(torch.square(targets - pred_targets))
    mae = torch.mean(torch.abs(targets - pred_targets))    
    
    return targets.detach().cpu().numpy(), pred_targets.detach().cpu().numpy()


# -------------------------------------------
# testing

model.eval()
test_targets = []
test_pred_targets = []
for batch in tqdm(test_loader):
    with torch.no_grad():
        target, pred_target = loop(
            model, 
            batch, 
            property_type = property_type, 
        )
    test_targets.append(target)
    test_pred_targets.append(pred_target)
    
test_targets = np.concatenate(test_targets)
test_pred_targets = np.concatenate(test_pred_targets)

test_results = pd.DataFrame()
test_results['Name_int'] = test_dataframe.Name_int
test_results['targets'] = test_targets
test_results['predictions'] = test_pred_targets

test_results=test_results.groupby('Name_int').apply(lambda x: x.mean())

test_MAE = np.mean(np.abs(np.array(test_results['targets']) - np.array(test_results['predictions'])))
test_R2 = np.corrcoef(np.array(test_results['targets']), np.array(test_results['predictions']))[0][1] ** 2

print('MAE:', test_MAE, 'R2:', test_R2)
    
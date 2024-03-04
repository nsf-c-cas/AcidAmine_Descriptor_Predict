import torch
import torch_geometric
import numpy as np
import pandas as pd

import rdkit
import rdkit.Chem
from rdkit import Chem
import rdkit.Chem.AllChem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdMolAlign

from tqdm import tqdm
from copy import deepcopy
import pickle
import random
import re
import os
import subprocess
import gzip
import shutil

from multiprocessing import Pool
from functools import partial

from datasets.dataset_2D import Dataset_2D
from models.models_2d import GNN2D

def get_sec_amine_idx(mol):    
    substructure = Chem.MolFromSmarts('[H]N([C])[C]')
    indexsall = mol.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found')
        return None
    
    indexsall = indexsall[0]
    
    for i in indexsall:
        if mol.GetAtomWithIdx(i).GetSymbol() == 'N':
            N1 = i
    C_ = []
    for nei in mol.GetAtomWithIdx(N1).GetNeighbors():
        if nei.GetSymbol() =='C':
            C_.append(nei.GetIdx())
        if nei.GetSymbol() =='H':
            H4 = nei.GetIdx()
            
    return N1, H4, *C_

def get_NH2_idx(mol):
    
    substructure = Chem.MolFromSmarts('[NH2]C')
    indexsall = mol.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found')
        return None
    
    h=[]
    for i in indexsall[0]:
        if mol.GetAtomWithIdx(i).GetSymbol() == 'N':
            N1 = i
    for nei in mol.GetAtomWithIdx(N1).GetNeighbors():
        if nei.GetSymbol() =='C':
            C2 = nei.GetIdx()
        if nei.GetSymbol() =='H':
            h.append(nei.GetIdx())
            
    H3 = h[0]
    H4 = h[1]
            
    return N1, C2, H3, H4

def get_COOH_idx(mol):
    substructure = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    indexsall = mol.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found')
        return None
    
    o_append=[]
    for i, num in enumerate(range(mol.GetNumAtoms())):
        if i in indexsall[0]:
            if mol.GetAtomWithIdx(i).GetSymbol() == 'C':
                C1 = i
            if mol.GetAtomWithIdx(i).GetSymbol() == 'O':
                o_append.append(i)
    for o in o_append:
        if mol.GetBondBetweenAtoms(o,C1).GetBondType() == Chem.rdchem.BondType.SINGLE:
            O3 = o
        if mol.GetBondBetweenAtoms(o,C1).GetBondType() == Chem.rdchem.BondType.DOUBLE:
            O2 = o
    for nei in mol.GetAtomWithIdx(C1).GetNeighbors():
        if nei.GetSymbol() !='O':
            C4 = nei.GetIdx()
    for nei in mol.GetAtomWithIdx(O3).GetNeighbors():
        if nei.GetSymbol() =='H':
            H5 = nei.GetIdx()
            
    return C1, O2, O3, C4, H5


def generate_dataframe(smiles_list, molecule_type, N_cpus = 8):
    assert molecule_type in ['acids', 'pamine', 'samine']
    
    # create starting 3D rdkit mol objects from smiles
    canon_smiles = [rdkit.Chem.MolToSmiles(rdkit.Chem.MolFromSmiles(m)) for m in smiles_list]
    mols = [rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(m)) for m in canon_smiles]
        
    # set up dataframe to organize data
    test_dataframe = pd.DataFrame()
    test_dataframe['mols'] = [m for m in mols]
    test_dataframe['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in test_dataframe['mols']]
    test_dataframe['smiles'] = smiles_list
    test_dataframe['canon_smiles'] = [rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(m)) for m in test_dataframe['mols']]
    
    # get atomic indices of functional group
    if molecule_type == 'acids':
        atom_indices = [get_COOH_idx(m) for m in mols]
    elif molecule_type == 'pamine':
        atom_indices = [get_NH2_idx(m) for m in mols]
    elif molecule_type == 'samine':
        atom_indices = [get_sec_amine_idx(m) for m in mols]
    
    if molecule_type == 'acids':
        test_dataframe[['C1', 'O2', 'O3', 'C4', 'H5']] = np.array(atom_indices)
    elif molecule_type == 'pamine':
        test_dataframe[['N1', 'C2', 'H3', 'H4']] = np.array(atom_indices)
    elif molecule_type == 'samine':
        test_dataframe[['N1', 'H4', 'C1', 'C2']] = np.array(atom_indices)
        
    return test_dataframe
    
    
def load_model(molecule_type, property_type, select_property, property_aggregation, model_dictionary, device = torch.device("cpu")):
    
    pretrained_model = model_dictionary[(molecule_type, property_type, select_property, property_aggregation)]
    
    model = GNN2D(
        model_type = 'GIN',
        property_type = property_type, # atom, bond, mol 
        out_emb_channels = 128,
        hidden_channels = 128,
        atom_feature_dim = 53, # dimension of atom features
        bond_feature_dim = 14, # dimension of bond features
        out_channels = 1, # number of regression targets
        act = 'swish',
        device = 'cpu',
    )

    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)
    model.to(device)
    model.eval()
    
    return model


def make_predictions(test_dataframe, model, model_selection, atom_selection_dictionary, keep_explicit_hydrogens = False, remove_Hs_except_functional = False, device = torch.device("cpu")):
    
    molecule_type, property_type, property_name, property_aggregation = model_selection
    
    if property_type == 'atom':
        select_atom = atom_selection_dictionary[(molecule_type, property_type, property_name)]
    elif property_type == 'bond':
        select_bond = atom_selection_dictionary[(molecule_type, property_type, property_name)]

    if property_type == 'bond':
        test_dataframe['bond_atom_tuple'] = [tuple([int(i),int(j)]) for i,j in zip(test_dataframe[select_bond[0]], test_dataframe[select_bond[1]])]
    elif property_type == 'atom':
        test_dataframe['atom_index'] = [int(i) for i in test_dataframe[select_atom]]
    
    atom_ID_test = None
    if property_type == 'atom':
        atom_ID_test = np.array(list(test_dataframe.atom_index), dtype = int)
        
    bond_ID_1_test = None
    bond_ID_2_test = None
    if property_type == 'bond':
        bond_ID_1_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 0]
        bond_ID_2_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 1]

        
    if keep_explicit_hydrogens:
        mols_test = list(test_dataframe.mols)
    else:
        mols_test = list(test_dataframe.mols_noHs)
    
    test_dataset = Dataset_2D(
        property_type = property_type,
        mols = mols_test, 
        targets = None,
        atom_ID = atom_ID_test,
        bond_ID_1 = bond_ID_1_test,
        bond_ID_2 = bond_ID_2_test,
        remove_Hs_except_functional = remove_Hs_except_functional,
    )
    
    test_loader = torch_geometric.loader.DataLoader(
        dataset = test_dataset,
        batch_size = 100,
        shuffle = False,
    )
    
    predictions = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        batch_size = max(batch.batch).item() + 1
        
        with torch.no_grad():
            out, _ = model(
                        x = batch.atom_features, 
                        edge_index = batch.edge_index, 
                        edge_attr = batch.edge_attr,
                        batch = batch.batch,
                        select_atom_index = batch.atom_ID_index if property_type == 'atom' else None,
                        select_bond_start_atom_index = batch.bond_start_ID_index if property_type == 'bond' else None,
                        select_bond_end_atom_index = batch.bond_end_ID_index if property_type == 'bond' else None,
                    )

            pred = out.squeeze().detach().cpu().numpy()
            if batch_size == 1:
                predictions.append(np.array([pred]))
            else:
                predictions.append(pred)

    predictions = np.concatenate(predictions)
    
    return predictions

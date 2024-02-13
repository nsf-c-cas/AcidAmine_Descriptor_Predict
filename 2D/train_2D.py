# -------------------------------------------
# imports
# -------------------------------------------

import torch
import torch_geometric
from datasets.dataset_2D import Dataset_2D
from models.models_2d import GNN2D

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

# -------------------------------------------
# Define job parameters
# -------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("output_PATH", type=str)
parser.add_argument("job_name", type=str)
parser.add_argument("dataframe_PATH", type=str)

parser.add_argument("property_type", type=str) # 'bond', 'atom', 'mol'

parser.add_argument("keep_explicit_hydrogens", type = int) # 0 or 1 or 2 (to keep only functional Hs)

parser.add_argument("model_type", type=str) # 'GIN'
parser.add_argument("pretrained_path", type=str) # path to pretrained model, or str 'none'

parser.add_argument("batch_size", type = int) # 128
parser.add_argument("lr", type = float) # 0.0001
parser.add_argument("seed", type = int)  # 0

parser.add_argument("loss_type", type = str) # 'mse' or 'mae'

args = parser.parse_args()

# -------------------------------------------
# Define job variables
# -------------------------------------------

output_PATH = args.output_PATH + '/'
job_name = args.job_name + '/'
output_dir = output_PATH + job_name

dataframe_PATH = args.dataframe_PATH

property_type = args.property_type 

keep_explicit_hydrogens = bool(args.keep_explicit_hydrogens) # False or True
remove_Hs_except_functional = args.keep_explicit_hydrogens == 2 # assumes keep_explicit_hydrogens == True

model_type = args.model_type

pretrained_model = '' 
if args.pretrained_path not in ['none', ' ', '', None, 'None']:
    pretrained_model = f'{args.pretrained_path}'

batch_size = args.batch_size 
lr = args.lr 
loss_type = args.loss_type

seed = args.seed

# defaults
num_workers = 6
N_epochs = 1000

# -------------------------------------------
# setting up logging
# -------------------------------------------

if not os.path.exists(output_PATH):
    os.makedirs(output_PATH)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(output_dir + 'saved_models'):
    os.makedirs(output_dir + 'saved_models')

def logger(text, file = output_dir + 'log.txt'):
    with open(file, 'a') as f:
        f.write(text + '\n')

logger(f'job_id: {int(os.environ.get("SLURM_JOB_ID"))}')
logger(f'python command: {sys.argv}')
logger(f'args: {args}')
logger(f' ')

logger(f'property_type: {property_type}')

logger(f'keep_explicit_hydrogens: {keep_explicit_hydrogens}')
logger(f'remove_Hs_except_functional: {remove_Hs_except_functional}')

logger(f'model_type: {model_type}')
logger(f'pretrained_model: {pretrained_model}')

logger(f'num_workers: {num_workers}')
logger(f'batch_size: {batch_size}')
logger(f'lr: {lr}')
logger(f'N_epochs: {N_epochs}')
logger(f'seed: {seed}')


# -------------------------------------------
# loading dataframes (and splits)
# -------------------------------------------
"""
Feel free to change the following code to do any dataset pre-processing, if your datasets are not pre-generated in this manner.

Important:
This code assumes that train_dataframe, val_dataframe, and test_dataframe have the following columns:
    
    mols - rdkit mol object with explicit hydrogens included
    
    mols_noHs - rdkit mol object without explicit hydrogens
    
    y - value (float) of target property (e.g., for single-task regression). These are directly used for training, without any processing.

    (if property_type == 'atom')
        atom_index - index (int) of query atom. This index MUST be the same regardless of whether the mol or mol_noHs is used.
    
    (if property_type == 'bond')
        bond_atom_tuple - tuple of two atom indices (int) comprising the query bond. These indices MUST be the same regardless of whether mol or mol_noHs is used.
    
    split - 'train', 'val', or 'test'
    
    (optional)
    smiles - SMILES string of molecule
"""

df = pd.read_pickle(dataframe_PATH, 'gzip')

use_smiles = False 
# demonstration of how data can be formatted to fit the above requirements
if use_smiles:
    from datasets.dataset_2D import get_COOH_idx, get_NH2_idx, get_sec_amine_idx
    
    mols = []
    mols_noHs = []
    atom_index = []
    bond_atom_tuple = [] 
    for s in df.smiles:
        m = rdkit.Chem.MolFromSmiles(s)
        m = rdkit.Chem.RemoveHs(m)
        
        # guaranteeing canonical atom ordering, with hydrogens ordered after heavy atoms
        s_canon = rdkit.Chem.MolToSmiles(m)
        m = rdkit.Chem.MolFromSmiles(s_canon)
        m = rdkit.Chem.RemoveHs(m)
        mol = rdkit.Chem.AddHs(m)
        
        acid_substructure = rdkit.Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
        acid_indexsall = mol.GetSubstructMatches(acid_substructure)
        amine_substructure = rdkit.Chem.MolFromSmarts('[NH2]C')
        amine_indexsall = mol.GetSubstructMatches(amine_substructure)
        sec_amine_substructure = rdkit.Chem.MolFromSmarts('[H]N([C])[C]')
        sec_amine_indexsall = mol.GetSubstructMatches(sec_amine_substructure)
        if len(acid_indexsall) > 0:
            C1, O2, O3, C4, H5 = get_COOH_idx(mol)
        elif len(amine_indexsall) > 0:
            N1, C2, H3, H4 = get_NH2_idx(mol)
        elif len(sec_amine_indexsall) > 0:
            N1, H4, _, _ = get_sec_amine_idx(mol)
        
        if property_type == 'atom':
            # need to specify
            atom_type = ''
            assert atom_type != ''
            
            # acids 
            if atom_type == 'C1':
                atom_index.append(C1)
            if atom_type == 'O2':
                atom_index.append(O2)
            if atom_type == 'O3':
                atom_index.append(O3)
            if atom_type == 'C4':
                atom_index.append(C4)
            if atom_type == 'H5':
                atom_index.append(H5)
                
            # amines 
            if atom_type == 'N1':
                atom_index.append(N1)
            if atom_type == 'C2':
                atom_index.append(C2)
            if atom_type == 'H3':
                atom_index.append(H3)
            if atom_type == 'H4':
                atom_index.append(H4)
                            
            # secondary amines 
            if atom_type == 'N1':
                atom_index.append(N1)
            if atom_type == 'H4':
                atom_index.append(H4)
        
        if property_type == 'bond':
            # need to specify
            bond_type = ('','')
            assert bond_type != ('', '')
            
            # acids
            if bond_type == ('C1', 'C4'):
                bond_atom_tuple.append((C1, C4))
            if bond_type == ('C1', 'O2'):
                bond_atom_tuple.append((C1, O2))
            
            # amines
            if bond_type == ('N1', 'C2'):
                bond_atom_tuple.append((N1, C2))
        
    df['mols'] = mols
    df['mols_noHs'] = mols_noHs
    
    if property_type == 'atom':
        df['atom_index'] = atom_index
        
    if property_type == 'bond':
        df['bond_atom_tuple'] = bond_atom_tuple


train_dataframe = df[df.split == 'train'].reset_index(drop = True)
val_dataframe = df[df.split == 'valid'].reset_index(drop = True)
test_dataframe = df[df.split == 'test'].reset_index(drop = True)


# -------------------------------------------
# initializing datasets and dataloaders
# -------------------------------------------

random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)
device = "cpu"

atom_ID_train, atom_ID_val, atom_ID_test = None, None, None
if property_type == 'atom':
    atom_ID_train = np.array(list(train_dataframe.atom_index), dtype = int)
    atom_ID_val = np.array(list(val_dataframe.atom_index), dtype = int)
    atom_ID_test = np.array(list(test_dataframe.atom_index), dtype = int)

bond_ID_1_train, bond_ID_2_train, bond_ID_1_val, bond_ID_2_val, bond_ID_1_test, bond_ID_2_test = None, None, None, None, None, None
if property_type == 'bond':
    bond_ID_1_train = np.array(list(train_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_train = np.array(list(train_dataframe.bond_atom_tuple), dtype = int)[:, 1]
    bond_ID_1_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 1]
    bond_ID_1_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_test = np.array(list(test_dataframe.bond_atom_tuple), dtype = int)[:, 1]


if keep_explicit_hydrogens:
    mols_train = list(train_dataframe.mols)
    mols_val = list(val_dataframe.mols)
    mols_test = list(test_dataframe.mols)
else:
    assert remove_Hs_except_functional == False
    mols_train = list(train_dataframe.mols_noHs)
    mols_val = list(val_dataframe.mols_noHs)
    mols_test = list(test_dataframe.mols_noHs)


train_dataset = Dataset_2D(
    property_type = property_type,
    mols = mols_train, 
    targets = list(train_dataframe['y']),
    atom_ID = atom_ID_train,
    bond_ID_1 = bond_ID_1_train,
    bond_ID_2 = bond_ID_2_train,
    remove_Hs_except_functional = remove_Hs_except_functional,
)
val_dataset = Dataset_2D(
    property_type = property_type,
    mols = mols_val, 
    targets = list(val_dataframe['y']),
    atom_ID = atom_ID_val,
    bond_ID_1 = bond_ID_1_val,
    bond_ID_2 = bond_ID_2_val,
    remove_Hs_except_functional = remove_Hs_except_functional,
)
test_dataset = Dataset_2D(
    property_type = property_type,
    mols = mols_test, 
    targets = list(test_dataframe['y']),
    atom_ID = atom_ID_test,
    bond_ID_1 = bond_ID_1_test,
    bond_ID_2 = bond_ID_2_test,
    remove_Hs_except_functional = remove_Hs_except_functional,
)


train_loader = torch_geometric.loader.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_workers,
)
val_loader = torch_geometric.loader.DataLoader(
    dataset = val_dataset,
    batch_size = 100,
    shuffle = False,
    num_workers = num_workers,
)
test_loader = torch_geometric.loader.DataLoader(
    dataset = test_dataset,
    batch_size = 100,
    shuffle = False,
    num_workers = num_workers,
)

# -------------------------------------------
# initializing model and optimizer
# -------------------------------------------

if model_type in ['GIN']:
    model = GNN2D(
        model_type = model_type,
        property_type = property_type, # atom, bond, mol 
        out_emb_channels = 128,
        hidden_channels = 128,
        atom_feature_dim = 53, # dimension of atom features
        bond_feature_dim = 14, # dimension of bond features
        out_channels = 1, # number of regression targets
        act = 'swish',
        device = 'cpu',
    )
else:
    raise Exception(f'model type <{model_type}> not implemented')

if pretrained_model != '':
    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)


# -------------------------------------------
# define training loop
# -------------------------------------------

def loop(model, data, training = True, property_type = 'bond', loss_type = 'mse'):
    
    if training:
        optimizer.zero_grad()
    
    data = data.to(device)
    batch_size = max(data.batch).item() + 1
        
    if model_type in ['GIN']:
        out, _ = model(
            x = data.atom_features, 
            edge_index = data.edge_index, 
            edge_attr = data.edge_attr,
            batch = data.batch,
            select_atom_index = data.atom_ID_index if property_type == 'atom' else None,
            select_bond_start_atom_index = data.bond_start_ID_index if property_type == 'bond' else None,
            select_bond_end_atom_index = data.bond_end_ID_index if property_type == 'bond' else None,
        )
        
    targets = data.targets
    pred_targets = out.squeeze()
    mse = torch.mean(torch.square(targets - pred_targets))
    mae = torch.mean(torch.abs(targets - pred_targets))   
    
    if loss_type == 'mse':
        backprop_loss = mse
    elif loss_type == 'mae':
        backprop_loss = mae
        
    if training:
        backprop_loss.backward()
        optimizer.step()
    
    return batch_size, backprop_loss.item(), mae.item(), targets.detach().cpu().numpy(), pred_targets.detach().cpu().numpy()


# -------------------------------------------
# training 
# -------------------------------------------

logger('starting to train')

epoch = 0

epoch_train_losses = []
epoch_val_losses = []
epoch_val_maes = []
epoch_val_R2s = []

best_mae = np.inf

while epoch < N_epochs:
    epoch += 1
    
    model.train()
    batch_losses = []
    batch_sizes = []
    for batch in train_loader:
        batch_size, loss, _, _, _ = loop(
            model, 
            batch,
            training = True,
            property_type = property_type,
            loss_type = loss_type,
        )
        batch_losses.append(loss)
        batch_sizes.append(batch_size)
        
    epoch_train_losses.append(
        np.sum(np.array(batch_losses) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes))
    )
    
    model.eval()
    batch_losses = []
    batch_maes = []
    batch_sizes = []
    val_targets = []
    val_pred_targets = []
    for batch in val_loader:
        with torch.no_grad():
            batch_size, loss,  mae, v_target, v_pred_target = loop(
                model, 
                batch, 
                training = False, 
                property_type = property_type, 
                loss_type = loss_type,
            )
        batch_losses.append(loss)
        batch_maes.append(mae)
        batch_sizes.append(batch_size)
        val_targets.append(v_target)
        val_pred_targets.append(v_pred_target)
    
    epoch_val_losses.append(np.sum(np.array(batch_losses) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    epoch_val_maes.append(np.sum(np.array(batch_maes) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    
    val_targets = np.concatenate(val_targets)
    val_pred_targets = np.concatenate(val_pred_targets)
    val_R2 = np.corrcoef(val_targets, val_pred_targets)[0][1] ** 2
    epoch_val_R2s.append(val_R2)
    
    
    logger(f'epoch: {epoch}, train_loss: {epoch_train_losses[-1]}, val_loss: {epoch_val_losses[-1]}, val_mea: {epoch_val_maes[-1]}, val_R2: {epoch_val_R2s[-1]}')
    
    
    test_this_epoch = False
    if epoch_val_maes[-1] < best_mae:
        test_this_epoch = True
        best_mae = epoch_val_maes[-1]
        logger(f'saving best model after epoch {epoch}. Best validation mae: {best_mae}')
        torch.save(model.state_dict(), output_dir + f'saved_models/model_best.pt')
        np.save(output_dir + 'training_losses.npy', np.array(epoch_train_losses)) # training loss curve
        np.save(output_dir + 'validation_losses.npy', np.array(epoch_val_losses)) # validation loss curve
        np.save(output_dir + 'validation_mae.npy', np.array(epoch_val_maes)) # validation mae curve
        np.save(output_dir + 'validation_R2.npy', np.array(epoch_val_R2s)) # validation r2 curve
    
    if epoch % 5 == 0:
        logger(f'checkpointing model at epoch {epoch}')
        torch.save(model.state_dict(), output_dir + f'saved_models/model_checkpoint.pt')
        np.save(output_dir + 'training_losses.npy', np.array(epoch_train_losses)) # training loss curve
        np.save(output_dir + 'validation_losses.npy', np.array(epoch_val_losses)) # validation loss curve
        np.save(output_dir + 'validation_mae.npy', np.array(epoch_val_maes)) # validation mae curve
        np.save(output_dir + 'validation_R2.npy', np.array(epoch_val_R2s)) # validation r2 curve
        
    
    # TESTING
    if test_this_epoch:
        
        model.eval()
        test_targets = []
        test_pred_targets = []
        for batch in test_loader:
            with torch.no_grad():
                _, _,  _, t_target, t_pred_target = loop(
                    model, 
                    batch, 
                    training = False, 
                    property_type = property_type, 
                    loss_type = loss_type,
                )
            test_targets.append(t_target)
            test_pred_targets.append(t_pred_target)
        
        test_targets = np.concatenate(test_targets)
        test_pred_targets = np.concatenate(test_pred_targets)
        
        test_mae = np.mean(np.abs(test_targets - test_pred_targets))
        test_R2 = np.corrcoef(test_targets, test_pred_targets)[0][1] ** 2
        
        logger(f'TESTING: MAE = {test_mae.round(5)}, R2 = {test_R2.round(5)}')
        
        # saving predictions for each molecule in test set
        np.save(f'{output_dir}/test_predictions.npy', np.stack((test_targets, test_pred_targets), axis = 0)) 

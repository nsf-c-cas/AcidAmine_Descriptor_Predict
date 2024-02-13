"""
# Example Python calls:

python train.py jobs_example_acid_C1_NBO_charge_Boltz/ data/3D_model_acid_rdkit_conformers.csv data/acid/atom/C1_NBO_charge_Boltz.csv atom 10 0 none 128 0.0001 0 0 1

python train.py jobs_example_acid_IR_freq_lowE/ data/3D_model_acid_rdkit_conformers.csv data/acid/bond/IR_freq_lowE.csv bond 10 0 none 128 0.0001 0 0 1

python train.py jobs_example_secamine_NBO_charge_H_max/ data/3D_model_secondaryamine_rdkit_conformers.csv data/secondary_amine/atom/NBO_charge_H_max.csv atom 20 2 none 128 0.0001 0 0 1

python train.py jobs_example_combinedamine_dipole_min/ data/3D_model_combinedamine_rdkit_conformers.csv data/combined_amine/mol/dipole_min.csv mol 20 0 none 128 0.0001 0 0 1

python train.py jobs_example_primaryamine_Sterimol_B5_lowE/ data/3D_model_primaryamine_rdkit_conformers.csv data/primary_amine/bond/Sterimol_B5_lowE.csv bond 20 0 none 128 0.0001 0 0 1

"""

print('importing modules...')

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

#-------------------------------------------
# Define job parameters
#-------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("output_dir", type=str) # directory containing output files, model checkpoints, etc.
parser.add_argument("conformer_data_file", type=str) # path to conformer dataset csv file
parser.add_argument("descriptor_data_file", type=str) # path to descriptor dataset csv file
parser.add_argument("property_type", type=str) # 'bond', 'atom', 'mol'
parser.add_argument("N_conformers", type=int) # 10 for acids, 20 for amines
parser.add_argument("keep_explicit_hydrogens", type = int) # 0 or 1 or 2
parser.add_argument("pretrained_path", type=str) # path to pretrained model, or str 'none'
parser.add_argument("batch_size", type = int) # 128
parser.add_argument("lr", type = float) # 0.0001
parser.add_argument("seed", type = int)  # 0
parser.add_argument("epoch", type = int) # 0
parser.add_argument("use_atom_features", type = int) # 1

args = parser.parse_args()

output_dir = args.output_dir + '/'

property_type = args.property_type # 'bond', 'atom', or 'mol'

N_conformers = args.N_conformers

keep_explicit_hydrogens = bool(args.keep_explicit_hydrogens) # False or True
remove_Hs_except_functional = args.keep_explicit_hydrogens == 2 # assumes keep_explicit_hydrogens == True

pretrained_model = '' if args.pretrained_path in ['', ' ', 'N', 'none', 'None', '0', 'false', 'False', '-1'] else args.pretrained_path
restart_from_checkpoint = args.pretrained_path == 'checkpoint'
if restart_from_checkpoint:
    pretrained_model = f'{output_dir}/saved_models/model_checkpoint.pt'  
    
batch_size = args.batch_size 
lr = args.lr 

num_workers = 4
N_epochs = 3000
save_every_epoch = 100

seed = args.seed # 0 

use_atom_features = bool(args.use_atom_features) # default should be True


#-------------------------------------------
# logging

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(output_dir + 'saved_models'):
    os.makedirs(output_dir + 'saved_models')

def logger(text, file = output_dir + 'log.txt'):
    print(text)
    with open(file, 'a') as f:
        f.write(text + '\n')

logger(f'python command: {sys.argv}')
logger(f'args: {args}')
logger(f' ')
logger(f' ')
logger(f'conformer_data_file: {args.conformer_data_file}')
logger(f'descriptor_data_file: {args.descriptor_data_file}')
logger(f' ')
logger(f'property_type: {property_type}')
logger(f'N_conformers: {N_conformers}')
logger(f'keep_explicit_hydrogens: {keep_explicit_hydrogens}')
logger(f'remove_Hs_except_functional: {remove_Hs_except_functional}')
logger(f'pretrained_model: {pretrained_model}')
logger(f'use_atom_features: {use_atom_features}')
logger(f'num_workers: {num_workers}')
logger(f'batch_size: {batch_size}')
logger(f'lr: {lr}')
logger(f'N_epochs: {N_epochs}')
logger(f'save_every_epoch: {save_every_epoch}')
logger(f'seed: {seed}')
logger(f'starting epoch: {args.epoch}')

#-------------------------------------------
# Loading training data (regression targets and input conformers)

if restart_from_checkpoint:
    logger('loading existing dataframes...')
    train_dataframe = pd.read_pickle(f'{output_dir}/train_df.pkl')
    val_dataframe = pd.read_pickle(f'{output_dir}/val_df.pkl')
    test_dataframe = pd.read_pickle(f'{output_dir}/test_df.pkl')

else:
    descriptor_df = pd.read_csv(args.descriptor_data_file, converters={"bond_atom_tuple": ast.literal_eval})

    conformers_df = pd.read_csv(args.conformer_data_file).reset_index(drop = True)
    conformers_df['mols'] = [rdkit.Chem.MolFromMolBlock(m, removeHs = False) for m in conformers_df.mol_block]
    conformers_df['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in conformers_df['mols']]

    merged_df = conformers_df.merge(descriptor_df, on = 'Name_int')
    
    # balancing dataset by sampling N_conformers conformers from each molecule, with possible upsampling of small ensembles
    groups = merged_df.groupby('Name_int', sort = False)
    sampled_indices = []
    for g in tqdm(groups):
        N_max = len(g[1])
        sample_N_times = max(N_conformers // N_max, 1) + 1
        sampled_conformers = []
        for _ in range(sample_N_times):
            sampled_conformers += list(np.random.choice(list(g[1].index), N_max, replace = False))
        assert len(sampled_conformers) >= N_conformers
        sampled_conformers = sampled_conformers[0:N_conformers]
        sampled_indices += (sampled_conformers)
    merged_df = merged_df.iloc[sampled_indices].reset_index(drop = True)
    
    train_dataframe = merged_df[merged_df.split == 'train'].reset_index(drop = True)
    val_dataframe = merged_df[merged_df.split == 'val'].reset_index(drop = True)
    test_dataframe = merged_df[merged_df.split == 'test'].reset_index(drop = True)
    
    logger('saving dataframes...')
    train_dataframe.to_pickle(output_dir + 'train_df.pkl')
    val_dataframe.to_pickle(output_dir + 'val_df.pkl')
    test_dataframe.to_pickle(output_dir + 'test_df.pkl')


logger("Sizes of training, validation, and test splits: ")
logger(f"{len(train_dataframe)}, {len(set(train_dataframe.Name_int))}")
logger(f"{len(val_dataframe)}, {len(set(val_dataframe.Name_int))}")
logger(f"{len(test_dataframe)}, {len(set(test_dataframe.Name_int))}")


#-------------------------------------------
# creating model, optimizer, dataloaders

random.seed(seed)
np.random.seed(seed = seed)
torch.manual_seed(seed)
device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

if property_type == 'atom':
    atom_ID_train = np.array(list(train_dataframe.atom_index), dtype = int)
    atom_ID_val = np.array(list(val_dataframe.atom_index), dtype = int)
else:
    atom_ID_train = None
    atom_ID_val = None
    
if property_type == 'bond':
    bond_ID_1_train = np.array(list(train_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_train = np.array(list(train_dataframe.bond_atom_tuple), dtype = int)[:, 1]
    bond_ID_1_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 0]
    bond_ID_2_val = np.array(list(val_dataframe.bond_atom_tuple), dtype = int)[:, 1]
else:
    bond_ID_1_train = None
    bond_ID_2_train = None
    bond_ID_1_val = None
    bond_ID_2_val = None
    
    
if keep_explicit_hydrogens:
    mols_train = list(train_dataframe.mols)
    mols_val = list(val_dataframe.mols)
else:
    mols_train = list(train_dataframe.mols_noHs)
    mols_val = list(val_dataframe.mols_noHs)

train_dataset = Dataset_3D(
    property_type = property_type,
    mols = mols_train, 
    mol_types = list(train_dataframe['mol_type']),
    targets = list(train_dataframe['y']),
    ligand_ID = np.array(train_dataframe['Name_int']),    
    atom_ID = atom_ID_train,
    bond_ID_1 = bond_ID_1_train,
    bond_ID_2 = bond_ID_2_train,
    remove_Hs_except_functional = remove_Hs_except_functional,
)
val_dataset = Dataset_3D(
    property_type = property_type,
    mols = mols_val, 
    mol_types = list(val_dataframe['mol_type']),
    targets = list(val_dataframe['y']),
    ligand_ID = np.array(val_dataframe['Name_int']),
    atom_ID = atom_ID_val,
    bond_ID_1 = bond_ID_1_val,
    bond_ID_2 = bond_ID_2_val,
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

example_data = val_dataset[0]
atom_feature_dim = int(example_data.atom_features.shape[-1]) # 53

model = DimeNetPlusPlus(
    property_type = property_type, 
    use_atom_features = use_atom_features, 
    atom_feature_dim = atom_feature_dim if use_atom_features else 1, 
)
if pretrained_model != '':
    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)

model.to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr)

#-------------------------------------------
# training loop definition

def loop(model, batch, training = True, property_type = 'bond'):
    
    if training:
        optimizer.zero_grad()
    
    batch = batch.to(device)
    batch_size = max(batch.batch).item() + 1
    
    ligand_batch_IDs = torch.unique_consecutive(batch.ligand_ID, return_inverse = True)[1]
    
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
    backprop_loss = mse_loss
    mae = torch.mean(torch.abs(targets - pred_targets))    
        
    if training:
        backprop_loss.backward()
        optimizer.step()
    
    return batch_size, backprop_loss.item(), mae.item(), targets.detach().cpu().numpy(), pred_targets.detach().cpu().numpy()


#-------------------------------------------
# training 

logger('starting to train')

epoch = args.epoch # 0

if restart_from_checkpoint:
    try:
        epoch_train_losses = list(np.load(output_dir + 'training_losses.npy'))
        epoch_val_losses = list(np.load(output_dir + 'validation_losses.npy'))
        epoch_val_maes = list(np.load(output_dir + 'validation_mae.npy'))
        epoch_val_R2s = list(np.load(output_dir + 'validation_R2.npy'))
    except:
        epoch_train_losses = []
        epoch_val_losses = []
        epoch_val_active_class_losses = []
        epoch_val_maes = []
        epoch_val_R2s = []
else:
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_active_class_losses = []
    epoch_val_maes = []
    epoch_val_R2s = []

best_mae = np.inf if (len(epoch_val_maes) == 0) else np.min(np.array(epoch_val_maes))

while epoch < N_epochs:
    epoch += 1
    
    model.train()
    batch_losses = []
    batch_sizes = []
    for batch in tqdm(train_loader, total = len(train_loader)):
        batch_size, loss, _, _, _ = loop(
            model, 
            batch,
            training = True,
            property_type = property_type,
        )
        batch_losses.append(loss)
        batch_sizes.append(batch_size)
    epoch_train_losses.append(np.sum(np.array(batch_losses) * np.array(batch_sizes)) / np.sum(np.array(batch_sizes)))
    
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
    
    logger(f'epoch: {epoch}, train_loss: {epoch_train_losses[-1]}, val_loss: {epoch_val_losses[-1]}, val_mae: {epoch_val_maes[-1]}, val_R2: {epoch_val_R2s[-1]}')
    
    if epoch_val_maes[-1] < best_mae:
        best_mae = epoch_val_maes[-1]
        logger(f'saving best model after epoch {epoch}. Best validation mae: {best_mae}')
        torch.save(model.state_dict(), output_dir + f'saved_models/model_best.pt')
        np.save(output_dir + 'training_losses.npy', np.array(epoch_train_losses))
        np.save(output_dir + 'validation_losses.npy', np.array(epoch_val_losses))
        np.save(output_dir + 'validation_mae.npy', np.array(epoch_val_maes))
        np.save(output_dir + 'validation_R2.npy', np.array(epoch_val_R2s))
    
    if epoch % 5 == 0:
        logger(f'checkpointing model at epoch {epoch}')
        torch.save(model.state_dict(), output_dir + f'saved_models/model_checkpoint.pt')
        np.save(output_dir + 'training_losses.npy', np.array(epoch_train_losses))
        np.save(output_dir + 'validation_losses.npy', np.array(epoch_val_losses))
        np.save(output_dir + 'validation_mae.npy', np.array(epoch_val_maes))
        np.save(output_dir + 'validation_R2.npy', np.array(epoch_val_R2s))

        
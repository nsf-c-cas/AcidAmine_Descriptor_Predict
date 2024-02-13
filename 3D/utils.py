import torch
import torch_geometric
import numpy as np
import pandas as pd

import rdkit
import rdkit.Chem
from rdkit import Chem
import rdkit.Chem.AllChem
from rdkit.Chem import rdMolAlign
from rdkit.ML.Cluster import Butina

from tqdm import tqdm
from copy import deepcopy

from multiprocessing import Pool
from functools import partial

from models.dimenetpp import *
from datasets.dataset_3D import *

def get_sec_amine_idx(mol, has_Hs = True):
    f"""
    Returns atom indices for secondary amine substructure in an RDKit mol object.
    Secondary amines are defined by the SMARTS pattern "[H]N([C])[C]"
    
    Notes: 
        The RDKit mol object should have explicit hydrogens. If not, hydrogens will be added, but this is not recommended.
        The RDKit mol object is assumed to have only 1 match for the SMARTS pattern.
    
    Args:
        mol - RDKit mol object containing exactly 1 "[H]N([C])[C]" substructure
        has_Hs - Whether or not the molecule has explicit hydrogens. 
    
    Returns:
        Tuple containing atom indices (N1, H4, C1, C2)
    """
    if not has_Hs:
        mol = rdkit.Chem.AddHs(mol)
        
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

def get_COOH_idx(mol, has_Hs = True):
    f"""
    Returns atom indices for carboxylic acid substructure in an RDKit mol object.
    Acids are defined by the SMARTS pattern "[CX3](=O)[OX2H1]"
    
    Notes: 
        The RDKit mol object should have explicit hydrogens. If not, hydrogens will be added, but this is not recommended.
        The RDKit mol object is assumed to have only 1 match for the SMARTS pattern.
    
    Args:
        mol - RDKit mol object containing exactly 1 "[CX3](=O)[OX2H1]" substructure
        has_Hs - Whether or not the molecule has explicit hydrogens. 
    
    Returns:
        Tuple containing atom indices (C1, O2, O3, C4, H5)
    """
    if not has_Hs:
        mol = rdkit.Chem.AddHs(mol)
    
    substructure = rdkit.Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
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
        if mol.GetBondBetweenAtoms(o,C1).GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE:
            O3 = o
        if mol.GetBondBetweenAtoms(o,C1).GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE:
            O2 = o
    for nei in mol.GetAtomWithIdx(C1).GetNeighbors():
        if nei.GetSymbol() =='C':
            C4 = nei.GetIdx()
    for nei in mol.GetAtomWithIdx(O3).GetNeighbors():
        if nei.GetSymbol() =='H':
            H5 = nei.GetIdx()
            
    return C1, O2, O3, C4, H5

def get_NH2_idx(mol, has_Hs = True):
    f"""
    Returns atom indices for primary amine substructure in an RDKit mol object.
    Primary amines are defined by the SMARTS pattern "[NH2]C"
    
    Notes: 
        The RDKit mol object should have explicit hydrogens. If not, hydrogens will be added, but this is not recommended.
        The RDKit mol object is assumed to have only 1 match for the SMARTS pattern.
    
    Args:
        mol - RDKit mol object containing exactly 1 "[NH2]C" substructure
        has_Hs - Whether or not the molecule has explicit hydrogens. 
    
    Returns:
        Tuple containing atom indices (N1, C2, H3, H4)
    """
        
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
    H3, H4 = h[0], h[1]
    
    return N1, C2, H3, H4


def embed_mol_ensemble(template_mol, numConfs=100, numThreads = 4):
    f"""
    Generates an MMFF94-optimized conformer ensemble of a given molecule, embedding conformers with ETKDG 
    and optimizing each with MMFF94.
    
    Args:
        template_mol - RDKit mol object with a (dummy) conformer, with explicit hydrogens.
        numConfs - Maximum number of conformers to be generated with ETKDG.
        numThreads - Number of cores to be used during conformer generation.
    
    Returns:
        mols - List of RDKit mol objects, each containing 1 conformer of the input molecule.
    """
    mol = deepcopy(template_mol)
    cids = rdkit.Chem.AllChem.EmbedMultipleConfs(mol, clearConfs=True, numConfs=numConfs, pruneRmsThresh = 0.25, maxAttempts = 50, numThreads = numThreads)
    for cid in cids: 
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    mols = [conf_to_mol(mol, c) for c in cids]
    return mols

def conf_to_mol(mol, conf_id):
    f"""
    Generates a new RDKit mol object for a specific conformer of a multi-conformer-containing RDKit mol object.
    
    Args:
        mol - RDKit mol object with multiple conformers.
        conf_id - Conformer ID of conformer to be converted into a new RDKit mol object.
    
    Returns:
        new_mol - RDKit mol object containing 1 conformer.
    """
    conf = mol.GetConformer(conf_id)
    new_mol = rdkit.Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(rdkit.Chem.Conformer(conf))
    return new_mol

def butina_clustering(ensemble, thresh = 0.2, N_max = 20):
    f"""
    Uses Butina Clustering to iteratively cluster a conformer ensemble into a maximum number of conformers,
    using increasing RMSD distance thresholds.
    
    Args:
        ensemble - A list of RDKit mol objects, each containing 1 conformer of a common molecule.
        thresh - Initial RMSD threshold used for clustering.
        N_max - Maximum number of cluster centroids to be returned in the clustered ensemble. 
    
    Returns:
        mols_clustered - List of RDKit mol objects containing the clustered conformers.
    """
    dists = []
    for i in range(len(ensemble)):
        for j in range(i):
            mol_i = rdkit.Chem.RemoveHs(deepcopy(ensemble[i]))
            mol_j = rdkit.Chem.RemoveHs(deepcopy(ensemble[j]))
            dists.append(rdMolAlign.GetBestRMS(mol_i, mol_j))
    
    clusts = Butina.ClusterData(dists, len(ensemble), thresh, isDistData=True, reordering=True)
    while len(clusts) > N_max:
        thresh += 0.1
        clusts = Butina.ClusterData(dists, len(ensemble), thresh, isDistData=True, reordering=True)

    select_confs = [c[0] for c in clusts] # picking centroids of each cluster
    mols_clustered = [ensemble[c] for c in select_confs]
    
    return mols_clustered


def canonicalize_mol(smiles):
    f"""
    Returns a canonical RDKit mol object (without hydrogens) for a given SMILES string.
    """
    return rdkit.Chem.MolFromSmiles(rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(rdkit.Chem.MolFromSmiles(smiles))))



def generate_rdkit_conformer_from_mol(mol, removeHs = False):
    f"""
    Generates a single conformer from an RDKit mol object, with ETKDG + MMFF94.
    
    Notes:
        Attempts to optimize the ETKDG-generated conformer with MMFF94. If MMFF94 optimization fails,
        then the un-optimized conformer (from ETKDG) is returned.
    
    Args:
        mol - RDKit mol object without a conformer.
        removeHs - Whether to remove hydrogens in the returned (3D) mol object
    
    Returns:
        mol - RDKit mol object with a conformer.
    """
    mol = rdkit.Chem.AddHs(mol)
    success = rdkit.Chem.AllChem.EmbedMolecule(mol, maxAttempts = 1000)
    if success != 0:
        print(f'failed to create rdkit conformer for {rdkit.Chem.MolToSmiles(mol)}')
        return None
    
    unoptimized_mol = deepcopy(mol)
    try:
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)
        mol.GetConformer()
    except:
        mol = unoptimized_mol
        print(f'warning: MMFF optimization failed for {rdkit.Chem.MolToSmiles(mol)}.')
        pass
    
    if removeHs:
        mol = rdkit.Chem.RemoveHs(mol)
    return mol


def generate_and_cluster_conformer_ensemble_from_smiles(smiles, N_confs = 20, N_cpus = 4):
    f"""
    Generates (with ETKDG), optimizes (with MMFF94), and clusters (with Butina clustering) 
    a conformer ensemble for a provided SMILES string.
    
    Args:
        smiles - SMILES string of molecule
        N_confs - maximum number of conformers in ensemble after clustering
        N_cpus - number of cores to use during conformer generation
    
    Returns:
        conformer_ensemble_clustered - List of RDKit mol objects, each containing 1 conformer.
    
    """
    template_3D_mol = generate_rdkit_conformer_from_mol(canonicalize_mol(smiles), removeHs = False)
    conformer_ensemble = embed_mol_ensemble(template_3D_mol, numConfs = N_confs*10, numThreads = N_cpus)
    conformer_ensemble_clustered = butina_clustering(conformer_ensemble, thresh = 0.2, N_max = N_confs)
    return conformer_ensemble_clustered
    

def generate_conformer_dataframe(smiles_list, molecule_type, N_cpus = 8, N_confs = 20):
    ligand_ID = list(range(len(smiles_list)))
    
    assert molecule_type in ['acid', 'amine', 'sec_amine']
    
    # create starting 3D rdkit mol objects from smiles
    template_3D_mols = [generate_rdkit_conformer_from_mol(canonicalize_mol(s), removeHs = False) for s in smiles_list]
    canon_smiles = [rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(m)) for m in template_3D_mols]
    
    # create conformer ensembles with MMFF
    print('generating and optimizing conformer ensembles...')
    ensembles = []
    for i in tqdm(range(len(template_3D_mols))):
        template_mol = template_3D_mols[i]    
        ensemble = embed_mol_ensemble(template_mol, numConfs = N_confs*5, numThreads = N_cpus)
        ensembles.append(ensemble)
    
    # cluster conformer ensembles by RMSD
    print('clustering conformer ensembles...')
    pool = Pool(N_cpus)
    clustered_ensembles = []
    for mols_clustered in tqdm(pool.imap(partial(butina_clustering, thresh = 0.2, N_max = N_confs), ensembles), total = len(ensembles)):
        clustered_ensembles.append(mols_clustered)
    pool.close()
        
        
    # set up dataframe to organize data
    test_dataframe = pd.DataFrame()
    test_dataframe['mols'] = [m for mols in clustered_ensembles for m in mols]
    test_dataframe['mols_noHs'] = [rdkit.Chem.RemoveHs(m) for m in test_dataframe['mols']]
    test_dataframe['smiles'] = [s for s_list in [[smiles_list[i]]*len(clustered_ensembles[i]) for i in range(len(smiles_list))] for s in s_list]
    test_dataframe['canon_smiles'] = [rdkit.Chem.MolToSmiles(rdkit.Chem.RemoveHs(m)) for m in test_dataframe['mols']]
    test_dataframe['mol_index'] = [j for j_list in [[ligand_ID[i]]*len(clustered_ensembles[i]) for i in range(len(smiles_list))] for j in j_list]
    
    
    # get atomic indices of functional group
    if molecule_type == 'acid':
        atom_indices = [get_COOH_idx(m) for m in template_3D_mols]
    elif molecule_type == 'amine':
        atom_indices = [get_NH2_idx(m) for m in template_3D_mols]
    elif molecule_type == 'sec_amine':
        atom_indices = [get_sec_amine_idx(m) for m in template_3D_mols]
    
    atom_indices_flatten = [a for a_list in [[atom_indices[i]]*len(clustered_ensembles[i]) for i in range(len(clustered_ensembles))] for a in a_list]
    atom_indices_flatten = np.array(atom_indices_flatten)
    
    if molecule_type == 'acid':
        test_dataframe[['C1', 'O2', 'O3', 'C4', 'H5']] = atom_indices_flatten
    elif molecule_type == 'amine':
        test_dataframe[['N1', 'C2', 'H3', 'H4']] = atom_indices_flatten
    elif molecule_type == 'sec_amine':
        test_dataframe[['N1', 'H4', 'C1', 'C2']] = atom_indices_flatten
    
    if molecule_type == 'acid':
        test_dataframe['mol_type'] = ['acid'] * len(test_dataframe)
    if molecule_type == 'amine':
        test_dataframe['mol_type'] = ['primary_amine'] * len(test_dataframe)
    if molecule_type == 'sec_amine':
        test_dataframe['mol_type'] = ['secondary_amine'] * len(test_dataframe)
        
    return test_dataframe
    
    
def load_model(molecule_type, property_type, select_property, property_aggregation, model_dictionary, device = torch.device("cpu")):
    
    assert molecule_type in ['acid', 'amine', 'sec_amine']
    
    pretrained_model = model_dictionary[(molecule_type, property_type, select_property, property_aggregation)]

    use_atom_features = True
    model = DimeNetPlusPlus(
        property_type = property_type, 
        use_atom_features = use_atom_features, 
        atom_feature_dim = 53 if use_atom_features else 1,
    )

    model.load_state_dict(torch.load(pretrained_model, map_location=next(model.parameters()).device), strict = True)
    model.to(device)
    model.eval()
    
    return model


def make_predictions(test_dataframe, model, model_selection, atom_selection_dictionary, keep_explicit_hydrogens = False, remove_Hs_except_functional = False, num_workers = 0, device = torch.device("cpu")):
    
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
    
    
    test_dataset = Dataset_3D(
        property_type = property_type,
        mols = mols_test, 
        mol_types = list(test_dataframe['mol_type']), 
        targets = None,
        ligand_ID = np.array(test_dataframe['mol_index']),
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
    
    
    predictions = []
    for batch in tqdm(test_loader):
        batch = batch.to(device)
        batch_size = max(batch.batch).item() + 1
        
        with torch.no_grad():
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
    
            pred = out[0].squeeze().detach().cpu().numpy()
            if batch_size == 1:
                predictions.append(np.array([pred]))
            else:
                predictions.append(pred)
        
    predictions = np.concatenate(predictions)
    
    return predictions

import torch
import torch_geometric
import numpy as np
import rdkit

import rdkit.Chem as Chem

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

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding

class Dataset_2D(torch_geometric.data.Dataset):
    def __init__(self, property_type, mols, targets = None, atom_ID = None, bond_ID_1 = None, bond_ID_2 = None, remove_Hs_except_functional = False):
        super(Dataset_2D, self).__init__()
        
        self.property_type = property_type # bond, atom, or mol
        self.mols = mols     
        self.atom_ID = atom_ID # numpy array containing atom IDs corresponding to each atom property in self.atom_targets
        self.bond_ID_1 = bond_ID_1 # numpy array containing the start atom index of the query bond 
        self.bond_ID_2 = bond_ID_2 # numpy array containing the end atom index of the query bond
        self.targets = targets
        self.remove_Hs_except_functional = remove_Hs_except_functional
    
    
    def get_bond_features(self, bond, mol):
        bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        features = one_hot_embedding(str(bond.GetBondType()), bondTypes) # dim=4+1
        features += [int(bond.GetIsConjugated())] # dim=1
        features += [int(bond.IsInRing())] # dim=1
        features += one_hot_embedding(bond.GetStereo(), list(range(6))) #dim=6+1
        return np.array(features, dtype = np.float32)

    
    def get_atom_features(self, atom, mol):
        features = one_hot_embedding(atom.GetSymbol(), ['C', 'O', 'H', 'N', 'F', 'S', 'Cl', 'Br', 'I', 'P', 'B', 'Si'])
        features += one_hot_embedding(atom.GetNumRadicalElectrons(), [0, 1, 2])
        chiral_types = [
            rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            rdkit.Chem.rdchem.ChiralType.CHI_OTHER,
        ]
        features += one_hot_embedding(str(atom.GetChiralTag()), chiral_types)
        features += one_hot_embedding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        features += [int(atom.GetIsAromatic())]
        
        # size of smallest ring to which the atom belongs, from 0 (no ring) up to 6 
        ring_info = mol.GetRingInfo()
        ring_associations = [len(ring) for ring in ring_info.AtomRings() if atom.GetIdx() in ring]
        min_ring_size = min(ring_associations) if len(ring_associations) > 0 else 0
        features += one_hot_embedding(min_ring_size, [0,1,2,3,4,5,6]) # if more than 6, the last code will be "hot"
        
        features += one_hot_embedding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
        features += one_hot_embedding(atom.GetTotalNumHs(includeNeighbors=True), [0,1,2,3,4,5,6])
        return features
        
    
    def featurize(self, mol):
        # Edge Index
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)
        
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        Z = np.array([atom.GetAtomicNum() for atom in atoms]) # Z
        
        atom_features = np.array([self.get_atom_features(atom, mol) for atom in atoms])
        
        bonds = [mol.GetBondBetweenAtoms(int(e[0]), int(e[1])) for e in edge_index.T]
        bond_features = np.array([self.get_bond_features(bond, mol) for bond in bonds])
        
        return edge_index, Z, atom_features, bond_features
        
    def process_mol(self, mol):
        edge_index, Z, atom_features, bond_features = self.featurize(mol)
        data = torch_geometric.data.Data(
            x = torch.as_tensor(Z).unsqueeze(1), 
            edge_index = torch.as_tensor(edge_index, dtype=torch.long),
        )
        data.atom_features = torch.as_tensor(atom_features, dtype = torch.float)
        data.edge_attr = torch.as_tensor(bond_features, dtype = torch.float)
        return data
    
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, key):
        mol = self.mols[key]
        
        targets = None
        if self.targets is not None:
            targets = torch.as_tensor(self.targets[key], dtype = torch.float)
        
        atom_ID = None
        if self.property_type == 'atom':
            atom_ID = self.atom_ID[key]

        bond_begin_atom_ID, bond_end_atom_ID = None, None
        if self.property_type == 'bond':
            bond_begin_atom_ID = self.bond_ID_1[key] 
            bond_end_atom_ID = self.bond_ID_2[key]
       
        data = self.process_mol(mol)
        
        data.targets = targets
        
        if self.remove_Hs_except_functional:
            acid_substructure = rdkit.Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
            acid_indexsall = mol.GetSubstructMatches(acid_substructure)
            amine_substructure = rdkit.Chem.MolFromSmarts('[NH2]C')
            amine_indexsall = mol.GetSubstructMatches(amine_substructure)
            sec_amine_substructure = rdkit.Chem.MolFromSmarts('[H]N([C])[C]')
            sec_amine_indexsall = mol.GetSubstructMatches(sec_amine_substructure)
            if len(acid_indexsall) > 0:
                functional_indices = get_COOH_idx(mol)
            elif len(amine_indexsall) > 0:
                functional_indices = get_NH2_idx(mol)
            elif len(sec_amine_indexsall) > 0:
                functional_indices = get_sec_amine_idx(mol)
            else:
                functional_indices = []
                
            Hs_indices = [a.GetIdx() for a in mol.GetAtoms() if (a.GetAtomicNum() == 1)]
            remove_Hs_indices = [i for i in Hs_indices if i not in functional_indices]
            subgraph_indices = list(set([a.GetIdx() for a in mol.GetAtoms()]) - set(remove_Hs_indices))
            
            x_subgraph = data.x[subgraph_indices]
            atom_features_subgraph = data.atom_features[subgraph_indices]
            
            edge_index_subgraph, _, edge_mask = torch_geometric.utils.subgraph(subgraph_indices, data.edge_index, relabel_nodes = True, return_edge_mask = True)
            
            # this assumes that all nodes have at least one edge (no isolated atoms in the molecule)
                #otherwise, we'll need to re-write torch_geometric.utils.subgraph to get a more robust atom_mapping
            atom_mapping = {int(i):int(j) for i,j in zip(data.edge_index[:, edge_mask][0,:], edge_index_subgraph[0,:])}
            
            data.x = x_subgraph
            data.atom_features = atom_features_subgraph
            data.edge_index = edge_index_subgraph
            data.edge_attr = data.edge_attr[edge_mask]
            
            atom_ID = atom_mapping[int(atom_ID)] if atom_ID is not None else None
            bond_begin_atom_ID = atom_mapping[int(bond_begin_atom_ID)] if bond_begin_atom_ID is not None else None
            bond_end_atom_ID = atom_mapping[int(bond_end_atom_ID)] if bond_end_atom_ID is not None else None
        
        
        # "_index" signals to torch_geometric to properly increment the atom index within each batch
        data.atom_ID_index = torch.as_tensor(atom_ID, dtype = torch.long) if atom_ID is not None else None 
        data.bond_start_ID_index = torch.as_tensor(bond_begin_atom_ID, dtype = torch.long) if bond_begin_atom_ID is not None else None 
        data.bond_end_ID_index = torch.as_tensor(bond_end_atom_ID, dtype = torch.long) if bond_end_atom_ID is not None else None 

        return data
    
    def len(self):
        return self.__len__()
    
    def get(self, key):
        return self.__getitem__(key)
    
    def getitem(self, key):
        return self.__getitem__(key)
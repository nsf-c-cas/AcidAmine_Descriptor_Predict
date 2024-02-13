import torch
import torch_geometric
import numpy as np
import rdkit
import rdkit.Chem as Chem

def get_sec_amine_idx(mol):
    f"""
    Returns atom indices for secondary amine substructure in an RDKit mol object.
    Secondary amines are defined by the SMARTS pattern "[H]N([C])[C]"
    
    Notes: 
        The RDKit mol object is assumed to have explicit hydrogens.
        The RDKit mol object is assumed to have only 1 match for the SMARTS pattern.
    
    Args:
        mol - RDKit mol object containing exactly 1 "[H]N([C])[C]" substructure
    
    Returns:
        Tuple containing atom indices (N1, H4, C1, C2)
    """
    substructure = Chem.MolFromSmarts('[H]N([C])[C]')
    indexsall = mol.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found', rdkit.Chem.MolToSmiles(mol))
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

def get_COOH_idx(mol):
    f"""
    Returns atom indices for carboxylic acid substructure in an RDKit mol object.
    Acids are defined by the SMARTS pattern "[CX3](=O)[OX2H1]"
    
    Notes: 
        The RDKit mol object is assumed to have explicit hydrogens.
        The RDKit mol object is assumed to have only 1 match for the SMARTS pattern.
    
    Args:
        mol - RDKit mol object containing exactly 1 "[CX3](=O)[OX2H1]" substructure
    
    Returns:
        Tuple containing atom indices (C1, O2, O3, C4, H5)
    """
    substructure = rdkit.Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
    indexsall = mol.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found', rdkit.Chem.MolToSmiles(mol))
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

def get_NH2_idx(mol):
    f"""
    Returns atom indices for primary amine substructure in an RDKit mol object.
    Primary amines are defined by the SMARTS pattern "[NH2]C"
    
    Notes: 
        The RDKit mol object is assumed to have explicit hydrogens.
        The RDKit mol object is assumed to have only 1 match for the SMARTS pattern.
    
    Args:
        mol - RDKit mol object containing exactly 1 "[NH2]C" substructure
    
    Returns:
        Tuple containing atom indices (N1, C2, H3, H4)
    """
        
    substructure = Chem.MolFromSmarts('[NH2]C')
    indexsall = mol.GetSubstructMatches(substructure)
    
    if len(indexsall) > 1:
        print('error: multiple matches found', rdkit.Chem.MolToSmiles(mol))
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


def one_hot_embedding(value, options):
    f"""
    Creates a one-hot embedding from a list of categorical variables.
    
    Examples:
        Inputs:
            value = 'apple'
            options = ['orange', 'apple', 'pear']
        Output:
            embedding == [0, 1, 0, 0]
            
        Inputs:
            value = 'pineapple' # not in options
            options = ['orange', 'apple', 'pear']
        Output:
            embedding == [0, 0, 0, 1]

    """
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding


class Dataset_3D(torch_geometric.data.Dataset):
    def __init__(self, property_type, mols, mol_types, targets = None, ligand_ID = None, atom_ID = None, bond_ID_1 = None, bond_ID_2 = None, remove_Hs_except_functional = False):
        super(Dataset_3D, self).__init__()
        
        self.property_type = property_type # bond, atom, or mol
        
        self.mols = mols
        self.mol_types = mol_types # 'acid', 'primary_amine', or 'secondary_amine' for each molecule
        
        self.ligand_ID = ligand_ID
        
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
        
        #chiral_types = list(rdkit.Chem.rdchem.ChiralType.names.values())
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
        if isinstance(mol, rdkit.Chem.rdchem.Conformer):
            mol = mol.GetOwningMol()
            conformer = mol
        elif isinstance(mol, rdkit.Chem.rdchem.Mol):
            mol = mol
            conformer = mol.GetConformer()
        
        # Edge Index
        adj = rdkit.Chem.GetAdjacencyMatrix(mol)
        adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
        array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
        edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
        edge_index[:, ::2] = array_adj
        edge_index[:, 1::2] = np.flipud(array_adj)
        
        atoms = rdkit.Chem.rdchem.Mol.GetAtoms(mol)
        Z = np.array([atom.GetAtomicNum() for atom in atoms]) # Z
        positions = np.array([conformer.GetAtomPosition(atom.GetIdx()) for atom in atoms]) # xyz positions
        
        atom_features = np.array([self.get_atom_features(atom, mol) for atom in atoms])
        
        bonds = [mol.GetBondBetweenAtoms(int(e[0]), int(e[1])) for e in edge_index.T]
        bond_features = np.array([self.get_bond_features(bond, mol) for bond in bonds])
        
        return edge_index, Z, positions, atom_features, bond_features # edge_index, Z, pos
        
    def process_mol(self, mol):
        edge_index, Z, pos, atom_features, bond_features = self.featurize(mol)
        data = torch_geometric.data.Data(
            x = torch.as_tensor(Z).unsqueeze(1), 
            edge_index = torch.as_tensor(edge_index, dtype=torch.long),
        )
        data.pos = torch.as_tensor(pos, dtype = torch.float)
        data.atom_features = torch.as_tensor(atom_features, dtype = torch.float)
        data.edge_attr = torch.as_tensor(bond_features, dtype = torch.float)
        return data
    
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self, key):
        mol = self.mols[key]
        mol_type = self.mol_types[key]
        
        lig_ID = None
        if self.ligand_ID is not None:
            lig_ID = torch.as_tensor(self.ligand_ID[key], dtype = torch.long)

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
        data.ligand_ID = lig_ID
        
        if self.remove_Hs_except_functional:
            
            functional_indices = []
            if mol_type == 'acid':
                acid_substructure = rdkit.Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
                acid_indexsall = mol.GetSubstructMatches(acid_substructure)
                if (len(acid_indexsall) > 1):
                    raise Exception(f'WARNING: multiple acid functional groups detected: {rdkit.Chem.MolToSmiles(mol)}')
                functional_indices = get_COOH_idx(mol)
            elif mol_type == 'primary_amine':
                amine_substructure = rdkit.Chem.MolFromSmarts('[NH2]C')
                amine_indexsall = mol.GetSubstructMatches(amine_substructure)
                if (len(amine_indexsall) > 1):
                    raise Exception(f'WARNING: multiple primary functional groups detected: {rdkit.Chem.MolToSmiles(mol)}')
                functional_indices = get_NH2_idx(mol)
            elif mol_type == 'secondary_amine':
                sec_amine_substructure = rdkit.Chem.MolFromSmarts('[H]N([C])[C]')
                sec_amine_indexsall = mol.GetSubstructMatches(sec_amine_substructure)
                if (len(sec_amine_indexsall) > 1):
                    raise Exception(f'WARNING: multiple secondary functional groups detected: {rdkit.Chem.MolToSmiles(mol)}')
                functional_indices = get_sec_amine_idx(mol)

            
            Hs_indices = [a.GetIdx() for a in mol.GetAtoms() if (a.GetAtomicNum() == 1)]
            remove_Hs_indices = [i for i in Hs_indices if i not in functional_indices]
            subgraph_indices = list(set([a.GetIdx() for a in mol.GetAtoms()]) - set(remove_Hs_indices))
            
            x_subgraph = data.x[subgraph_indices]
            pos_subgraph = data.pos[subgraph_indices]
            atom_features_subgraph = data.atom_features[subgraph_indices]
            
            # we may not even need to do this if we don't use the molecular graph during training/inference
            edge_index_subgraph, _, edge_mask = torch_geometric.utils.subgraph(subgraph_indices, data.edge_index, relabel_nodes = True, return_edge_mask = True)
            
            # this assumes that all nodes have at least one edge (no isolated atoms in the molecule)
                #otherwise, we'll need to re-write torch_geometric.utils.subgraph to get a more robust atom_mapping
            atom_mapping = {int(i):int(j) for i,j in zip(data.edge_index[:, edge_mask][0,:], edge_index_subgraph[0,:])}
            
            data.x = x_subgraph
            data.pos = pos_subgraph
            data.atom_features = atom_features_subgraph
            data.edge_index = edge_index_subgraph
            data.edge_attr = data.edge_attr[edge_mask]
            
            atom_ID = atom_mapping[int(atom_ID)] if atom_ID is not None else None
            bond_begin_atom_ID = atom_mapping[int(bond_begin_atom_ID)] if bond_begin_atom_ID is not None else None
            bond_end_atom_ID = atom_mapping[int(bond_end_atom_ID)] if bond_end_atom_ID is not None else None
        
        
        # "index" signals to torch_geometric to add to the atom index
        data.atom_ID_index = torch.as_tensor(atom_ID, dtype = torch.long) if atom_ID is not None else None 
        data.bond_start_ID_index = torch.as_tensor(bond_begin_atom_ID, dtype = torch.long) if bond_begin_atom_ID is not None else None 
        data.bond_end_ID_index = torch.as_tensor(bond_end_atom_ID, dtype = torch.long) if bond_end_atom_ID is not None else None 

        return data
    
    def len(self):
        return self.__len__()
    
    def get(self, key):
        return self.__getitem__(key)
        
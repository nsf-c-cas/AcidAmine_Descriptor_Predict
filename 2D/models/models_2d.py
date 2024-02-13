import torch
from torch import Tensor
from torch.nn import Linear

import torch_scatter
from torch_scatter import scatter

#from torch_geometric.nn.resolver import activation_resolver
from .models_2D.resolver import activation_resolver

from typing import Callable, Optional, Tuple, Union

class GNN2D(torch.nn.Module):
    
    def __init__(
        self, 
        model_type = 'GIN',
        property_type = 'bond', # atom, bond, mol 
        out_emb_channels: int = 128,
        hidden_channels: int = 128,
        atom_feature_dim: int = 53,
        bond_feature_dim: int = 14,
        out_channels: int = 1,
        act: Union[str, Callable] = 'swish',
        device = 'cpu',
    ):
        
        super().__init__()
        
        self.model_type = model_type
        self.property_type = property_type
        self.out_emb_channels = out_emb_channels
        self.hidden_channels = hidden_channels
        self.atom_feature_dim = atom_feature_dim
        self.bond_feature_dim = bond_feature_dim
        self.out_channels = out_channels
        self.act = act        
        self.device = device
        
        assert self.out_emb_channels == self.hidden_channels 
                
        if self.property_type in ['bond', 'atom']:
            self.property_predictor = MLP(
                out_emb_channels*2, 
                hidden_channels, 
                out_channels, 
                activation_resolver(self.act), 
                N_hidden_layers = 2,
            ) 
        if self.property_type == 'mol':
            self.property_predictor = MLP(
                out_emb_channels, 
                hidden_channels, 
                out_channels, 
                activation_resolver(self.act), 
                N_hidden_layers = 2,
            )
        if self.property_type == 'bond':
            self.pairwise_permutation_invariant_MLP = MLP(
                2*out_emb_channels, 
                hidden_channels, 
                out_emb_channels, 
                activation_resolver(self.act),
                N_hidden_layers = 1,
            ) 
                
        if model_type == 'GIN':
            from .models_2D.GIN import GIN
            self.model = GIN(
                out_emb_dim = self.out_emb_channels,
                hidden_dim = 128,
                num_layers = 4,
                atom_feature_dim = self.atom_feature_dim,
                bond_feature_dim = self.bond_feature_dim, 
                act = self.act,
            )
        else:
            raise Exception(f'model type {model_type} not implemented')
            
        self.to(torch.device(self.device))

    def forward(self, x, edge_index, edge_attr, batch, select_atom_index = None, select_bond_start_atom_index = None, select_bond_end_atom_index = None):
        
        if self.model_type in ['GIN']:
            h = self.model(
                x, 
                edge_index, 
                edge_attr, 
                batch,
            )
        else:
            raise Exception(f'model type {self.model_type} not implemented')
        
        if self.property_type == 'atom':
            out = self.predict_atom_properties(h, batch, select_atom_index)
        elif self.property_type == 'bond':
            out = self.predict_bond_properties(h, batch, select_bond_start_atom_index, select_bond_end_atom_index)
        elif self.property_type == 'mol':
            out = self.predict_mol_properties(h, batch)
        
        return out, h

    
    def predict_atom_properties(self, h, batch, select_atom_index, ensemble_batch = None):
        h_select = h[select_atom_index] # subselecting only certain atoms
        h_agg = torch_scatter.scatter_add(h, batch, dim = 0) # sum pooling over all atom embeddings to get molecule-level embedding
        h_out = torch.cat([h_select, h_agg], dim = 1)
        out = self.property_predictor(h_out)
        return out
    
    def predict_bond_properties(self, h, batch, select_bond_start_atom_index, select_bond_end_atom_index, ensemble_batch = None):        
        h_start = h[select_bond_start_atom_index] # selecting representations of atoms that start each bond
        h_end = h[select_bond_end_atom_index] # selecting representations of atoms that end each bond
        h_agg = torch_scatter.scatter_add(h, batch, dim = 0) # sum pooling over all atom embeddings to get molecule-level embedding
        h_bond = self.pairwise_permutation_invariant_MLP(torch.cat([h_start, h_end], dim = 1)) + self.pairwise_permutation_invariant_MLP(torch.cat([h_end, h_start], dim = 1))
        h_out = torch.cat([h_bond, h_agg], dim = 1)
        out = self.property_predictor(h_out)
        return out
    
    def predict_mol_properties(self, h, batch, ensemble_batch = None):
        h_out = torch_scatter.scatter_add(h, batch, dim = 0) # sum pooling over all atom embeddings to get molecule-level embedding
        out = self.property_predictor(h_out)
        return out


class MLP(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, act: Callable, N_hidden_layers = 2):
        super().__init__()
        self.act = act
        self.N_hidden_layers = N_hidden_layers
        
        if N_hidden_layers == 0:
            self.output_layer = Linear(input_channels, output_channels) # no activation
        else:
            self.input_layer = Linear(input_channels, hidden_channels) # this counts as 1 hidden layer
            self.lin_layers = torch.nn.ModuleList([
                Linear(hidden_channels, hidden_channels) for _ in range(N_hidden_layers - 1)
            ])
            self.output_layer = Linear(hidden_channels, output_channels)

    def forward(self, x: Tensor):
        if self.N_hidden_layers == 0:
            return self.output_layer(x)
        
        x = self.act(self.input_layer(x))
        for layer in self.lin_layers:
            x = self.act(layer(x))
        out = self.output_layer(x)
        
        return out

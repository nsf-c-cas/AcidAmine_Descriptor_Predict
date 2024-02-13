# adapted from https://github.com/SXKDZ/MARCEL/blob/main/benchmarks/models/model_2d.py

import torch
import torch.nn.functional as F

from torch.nn import Linear, Sequential, BatchNorm1d, Embedding, LayerNorm
from torch_geometric.nn import GINEConv, global_add_pool

#from torch_geometric.nn.resolver import activation_resolver
from .resolver import activation_resolver

import types
class Activation(torch.nn.Module):
    def __init__(self, act):
        self.act = act
        super().__init__()
    def forward(self, x):
        return self.act(x)

class GIN(torch.nn.Module):
    def __init__(
        self,
        out_emb_dim,
        hidden_dim,
        num_layers,
        atom_feature_dim,
        bond_feature_dim,
        act='swish',
    ):
        super().__init__()

        if type(activation_resolver(act)) == types.FunctionType:
            self.act = Activation(activation_resolver(act))
        else:
            self.act = activation_resolver(act)

        self.conv = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.conv.append(
                GINEConv(
                    Sequential(
                        Linear(hidden_dim, hidden_dim),
                        LayerNorm(hidden_dim), # BatchNorm1d(hidden_dim),
                        self.act,
                        Linear(hidden_dim, hidden_dim),
                        self.act,
                    )
                )
            )

        self.atom_encoder = torch.nn.Linear(atom_feature_dim, hidden_dim)
        self.bond_encoder = torch.nn.Linear(bond_feature_dim, hidden_dim)
        
        self.lin = torch.nn.Linear(hidden_dim, out_emb_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        
        for conv in self.conv:
            x = conv(x, edge_index, edge_attr)
        
        x = self.lin(x)
        x = self.act(x)
        
        return x
    
        #x = global_add_pool(x, batch)
        #return x
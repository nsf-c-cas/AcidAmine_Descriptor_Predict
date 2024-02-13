# copied from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/resolver.py
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/resolver.py

import inspect
from typing import Any, Optional, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

"""
from torch_geometric.nn.lr_scheduler import (
    ConstantWithWarmupLR,
    CosineWithWarmupLR,
    CosineWithWarmupRestartsLR,
    LinearWithWarmupLR,
    PolynomialWithWarmupLR,
)
"""

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # PyTorch < 2.0
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    
#from torch_geometric.resolver import normalize_string, resolver
import inspect
from typing import Any, Dict, List, Optional, Union

def normalize_string(s: str) -> str:
    return s.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolver(classes: List[Any], class_dict: Dict[str, Any],
             query: Union[Any, str], base_cls: Optional[Any],
             base_cls_repr: Optional[str], *args, **kwargs):

    if not isinstance(query, str):
        return query

    query_repr = normalize_string(query)
    if base_cls_repr is None:
        base_cls_repr = base_cls.__name__ if base_cls else ''
    base_cls_repr = normalize_string(base_cls_repr)

    for key_repr, cls in class_dict.items():
        if query_repr == key_repr:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                return obj
            return cls

    for cls in classes:
        cls_repr = normalize_string(cls.__name__)
        if query_repr in [cls_repr, cls_repr.replace(base_cls_repr, '')]:
            if inspect.isclass(cls):
                obj = cls(*args, **kwargs)
                return obj
            return cls

    choices = set(cls.__name__ for cls in classes) | set(class_dict.keys())
    raise ValueError(f"Could not resolve '{query}' among choices {choices}")
    
    
    
# Activation Resolver #########################################################


def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()


def activation_resolver(query: Union[Any, str] = 'relu', *args, **kwargs):
    base_cls = torch.nn.Module
    base_cls_repr = 'Act'
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


# Normalization Resolver ######################################################


def normalization_resolver(query: Union[Any, str], *args, **kwargs):
    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    base_cls_repr = 'Norm'
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)


# Aggregation Resolver ########################################################


def aggregation_resolver(query: Union[Any, str], *args, **kwargs):
    import torch_geometric.nn.aggr as aggr
    if isinstance(query, (list, tuple)):
        return aggr.MultiAggregation(query, *args, **kwargs)

    base_cls = aggr.Aggregation
    aggrs = [
        aggr for aggr in vars(aggr).values()
        if isinstance(aggr, type) and issubclass(aggr, base_cls)
    ]
    aggr_dict = {
        'add': aggr.SumAggregation,
    }
    return resolver(aggrs, aggr_dict, query, base_cls, None, *args, **kwargs)


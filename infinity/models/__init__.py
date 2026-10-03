import torch
from timm.loss import SoftTargetCrossEntropy

from timm.layers import DropPath

from .infinity3d_src2sos import Infinity3DSrc2Sos


def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy):  # no longer __repr__ DropPath with drop_prob
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'

DropPath.__repr__ = lambda self: f'{type(self).__name__}(...)'

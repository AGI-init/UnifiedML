# Copyright (c) Sam Lerman. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch
from torch import nn


class Ensemble(nn.Module):
    """
    "Ensembles" (stacks) multiple modules' outputs
    """
    def __init__(self, modules, dim=1):
        super().__init__()

        self.ensemble = nn.ModuleList([m if i == 0 or m != modules[i - 1]
                                       else deepcopy(m) for i, m in enumerate(modules)])
        self.dim = dim

    def forward(self, *x, **kwargs):
        out = [m(*x, **kwargs) for m in self.ensemble]

        return torch.stack(out, self.dim) if len(self) > 1 \
            else out[0].unsqueeze(self.dim)

    def __len__(self):
        return len(self.ensemble)

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.ensemble[0], key)

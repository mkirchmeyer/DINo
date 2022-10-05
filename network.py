# Copyright 2022 Yuan Yin & Matthieu Kirchmeyer

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import math
from torch.nn import init


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


nls = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'elu': partial(nn.ELU)}


class CodeBilinear(nn.Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, device=None, dtype=None) -> None:
        """
        x2T A + B x1
        x2: code, x1: spatial coordinates
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CodeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.A = Parameter(torch.empty(out_features, in2_features, **factory_kwargs))
        self.B = Parameter(torch.empty(out_features, in1_features, **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in1_features)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # input1: b, t, h, w, s, i
        # input2: b, t, s, j
        # W: o, i, j
        # B: o, i
        # A: o, j
        # bias: o
        res = 0
        
        bias_code = torch.einsum('btsj,oj->btso', input2, self.A)
        bias_code = bias_code.unsqueeze(2).unsqueeze(2)

        linear_trans_2 = torch.einsum('bthwsi,oi->bthwso', input1, self.B)
       
        res += linear_trans_2 
        res += bias_code
        res += self.bias
        return res
        
    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None)


class MLP(nn.Module):
    def __init__(self, code_size, hidden_size, nl='swish'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nls[nl](),
            nn.Linear(hidden_size, hidden_size),
            nls[nl](),
            nn.Linear(hidden_size, hidden_size),
            nls[nl](),
            nn.Linear(hidden_size, code_size),
        )

    def forward(self, x):
        return self.net(x)


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers):
        super().__init__()
        self.first = 3
        self.bilinear = nn.ModuleList(
            [CodeBilinear(in_size, code_size, hidden_size)] +
            [CodeBilinear(hidden_size, code_size, hidden_size) for _ in range(int(n_layers))]
        )
        self.output_bilinear = nn.Linear(hidden_size, out_size)
        return

    def forward(self, x, code):
        out = self.filters[0](x) * self.bilinear[0](x*0., code)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.bilinear[i](out, code)
        out = self.output_bilinear(out)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out, x


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    Adapted from https://github.com/boschresearch/multiplicative-filter-networks
    """
    def __init__(self, in_features, out_features, weight_scale):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return torch.cat([torch.sin(F.linear(x, self.weight * self.weight_scale)), torch.cos(F.linear(x, self.weight * self.weight_scale))], dim=-1)


class FourierNet(MFNBase):
    """
    Taken from https://github.com/boschresearch/multiplicative-filter-networks
    """
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, input_scale=256.0, **kwargs):
        super().__init__(in_size, hidden_size, code_size, out_size, n_layers)
        self.filters = nn.ModuleList(
                [FourierLayer(in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1)) for _ in range(n_layers + 1)])
    
    def get_filters_weight(self):
        weights = list()
        for ftr in self.filters:
            weights.append(ftr.weight)
        return torch.cat(weights)

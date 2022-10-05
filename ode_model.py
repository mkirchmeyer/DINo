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


from network import MLP, FourierNet
from torch import nn


class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        self.net = MLP(input_dim, hidden_c, nl='swish')

    def forward(self, t, u):
        return self.net(u)


class Decoder(nn.Module):
    def __init__(self, state_c, hidden_c, code_c, coord_dim, n_layers, **kwargs):
        super().__init__()
        self.state_c = state_c
        self.hidden_c = hidden_c
        self.coord_dim = coord_dim
        self.out_dim = 1
        self.code_dim = code_c
        self.net = FourierNet(self.coord_dim, self.hidden_c, self.code_dim, self.out_dim, n_layers, input_scale=64)

    def forward(self, x, codes=None):
        if codes is None:
            return self.net(x)
        return self.net(x, codes)

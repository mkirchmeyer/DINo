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


import torch
from torch.utils.data import Dataset
import math
from pde import ScalarField, CartesianGrid, MemoryStorage, PDE
from pde.pdes import WavePDE
import numpy as np
import os
import h5py

from netCDF4 import Dataset as netCDFDataset


def get_mgrid(sidelen, vmin=-1, vmax=1, dim=2):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    if isinstance(sidelen, int):
        tensors = tuple(dim * [torch.linspace(vmin, vmax, steps=sidelen)])
    elif isinstance(sidelen, (list, tuple)):
        if isinstance(vmin, (list, tuple)) and isinstance(vmax, (list, tuple)):
            tensors = tuple([torch.linspace(mi, ma, steps=l) for mi, ma, l in zip(vmin, vmax, sidelen)])
        else:
            tensors = tuple([torch.linspace(vmin, vmax, steps=l) for l in sidelen])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid


def get_mgrid_from_tensors(tensors):
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    return mgrid


##############
# Gray-Scott #
##############

class AbstractDataset(Dataset):
    def __init__(self, n_seq, n_seq_per_traj, size, t_horizon, dt, n_frames_train, buffer_shelve, group, scale=1, *args, **kwargs):
        super().__init__()
        self.n_seq = n_seq
        self.n_seq_per_traj = n_seq_per_traj
        self.size = size  # size of the 2D grid
        self.t_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt)  # number of iterations
        self.dt_eval = float(dt)
        assert group in ['train', 'train_eval', 'test', 'test_hr']
        self.group = group
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.buffer_shelve = buffer_shelve
        self.n_frames_train = n_frames_train
        self.scale = scale

    def _get_init_cond(self, index):
        raise NotImplementedError

    def _generate_trajectory(self, traj_id):
        raise NotImplementedError
    
    def _load_trajectory(self, traj_id):
        raise NotImplementedError

    def __getitem__(self, index):
        t = torch.arange(0, self.t_horizon, self.dt_eval).float()
        traj_id = index // self.n_seq_per_traj
        seq_id = index % self.n_seq_per_traj
        if self.buffer.get(f'{traj_id}') is None:
            if self.buffer_shelve is not None:
                if self.buffer_shelve.get(f'{traj_id}') is None:
                    self._generate_trajectory(traj_id)
                self.buffer[f'{traj_id}'] = self.buffer_shelve[f'{traj_id}']
            else:
                self.buffer[f'{traj_id}'] = self._load_trajectory(traj_id)
        data = self.buffer[f'{traj_id}']['data'][:, seq_id * self.n:(seq_id + 1) * self.n]  # (n_ch, T, H, W)
        data = torch.tensor(data).float().permute(1, 2, 3, 0)  # (T, H, W, n_ch)
        if self.group == 'train':
            data = data[:self.n_frames_train] / self.scale 
            t = t[:self.n_frames_train]

        return {
            'data': data, 
            't': t, 
            'traj': traj_id, 
            'index': index,
            'coords': self.coords,
        }

    def __len__(self):
        return self.n_seq

########
# Wave #
########


class WaveDataset(AbstractDataset):
    def __init__(self, param, coords='ang', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid = CartesianGrid([[-1., 1.]] * 2, self.size, periodic=True)
        self.eqs = WavePDE(**param)
        
        coords_list = []
        if 'ang' in coords:
            coords_list.append(get_mgrid(self.size, vmin=0, vmax=0.5, dim=2))
        if 'euc' in coords:
            grid = get_mgrid(self.size, -np.pi, np.pi, dim=2)
            phi = grid[..., 0]
            theta = grid[..., 1]
            R = 1
            r = 0.3
            coords_list.append(torch.stack([
                (R + r * torch.cos(phi)) * torch.cos(theta),
                (R + r * torch.cos(phi)) * torch.sin(theta),
                r * torch.sin(phi),
            ], dim=-1))

        self.coords = torch.cat(coords_list, dim=-1)
        self.coord_dim = self.coords.shape[-1]

    def _get_init_cond(self, index):
        np.random.seed(index if self.group != 'test' else self.max - index)
        r = 0.05 * np.random.rand() + 0.25
        x, y = np.meshgrid(np.linspace(-1, 1, self.size), np.linspace(-1, 1, self.size))
        dst = np.sqrt(x*x+y*y)
        # Calculating Gaussian array
        init_cond = np.exp(-(dst ** 2 / (2.0 * (r ** 2)))) * (2 + np.random.rand() * 2)
        init_cond = np.roll(init_cond, np.random.randint(self.size, size=2), axis=(0, 1))
        u = ScalarField(self.grid, init_cond)
        return self.eqs.get_initial_condition(u)

    def _generate_trajectory(self, traj_id):
        print(f'generating {traj_id}')
        storage = MemoryStorage()
        state = self._get_init_cond(traj_id)
        self.eqs.solve(state, t_range=self.t_horizon * self.n_seq_per_traj, dt=1e-3, tracker=storage.tracker(self.dt_eval))
        self.buffer_shelve[f'{traj_id}'] = {'data': np.stack(storage.data, axis=1)}


#################
# Navier Stokes #
#################


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None):
        self.dim = dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), torch.arange(start=-k_max, end=0, step=1)), 0)
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * ((4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


class NavierStokesDataset(AbstractDataset):
    def __init__(self, param, device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_eq = param
        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt = 1e-3
        self.device = device
        self.coords = get_mgrid(self.size, vmin=0, vmax=0.5, dim=2)
        self.coord_dim = self.coords.shape[-1]

    def navier_stokes_2d(self, w0, f, visc, T, delta_t, record_steps):
        # Grid size - must be power of 2
        N = w0.size()[-1]
        # Maximum frequency
        k_max = math.floor(N / 2.0)
        # Number of steps to final time
        steps = math.ceil(T / delta_t)
        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N, N))
        # Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N, N))
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                         torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        # Dealiasing mask
        dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
        sol_t = torch.zeros(record_steps, device=w0.device)
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, (N, N))
                # Record solution and time
                sol[..., c, 0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t
                c += 1
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap
            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N, N))
            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * temp
            v = torch.fft.ifftn(v, (N, N))
            # Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2 * math.pi * k_x * w_x.imag
            w_x.imag = 2 * math.pi * k_x * temp
            w_x = torch.fft.ifftn(w_x, (N, N))
            # Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2 * math.pi * k_y * w_y.imag
            w_y.imag = 2 * math.pi * k_y * temp
            w_y = torch.fft.ifftn(w_y, (N, N))
            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
            # Dealias
            F_h = dealias * F_h
            # Cranck-Nicholson update
            w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
                  (1.0 + 0.5 * delta_t * visc * lap)
            # Update real time (used only for recording)
            t += delta_t

        return sol, sol_t

    def _get_init_cond(self, index, start, end):
        print(f'generating {start}-{end-1} ICs')
        if self.buffer.get(f'init_cond_{index}') is None:
            w0s = []
            for i in range(start, end):
                torch.manual_seed(i if self.group != 'test' else self.max - i)
                w0 = self.sampler.sample().to(self.device)
                w0s.append(w0)
            w0 = torch.stack(w0s, 0)

            state, _ = self.navier_stokes_2d(w0, f=self.params_eq['f'].to(self.device), visc=self.params_eq['visc'], T=30,
                                             delta_t=self.dt, record_steps=20)
            init_cond = state[:, :, :, -1, 0].cpu()
            for i, ii in enumerate(range(start, end)):
                self.buffer[f'init_cond_{ii}'] = init_cond[i].numpy()
        else:
            init_cond = torch.from_numpy(torch.stack(self.buffer[f'init_cond_{i}'] for i in range(start, end)))

        return init_cond

    def _generate_trajectory(self, traj_id):
        batch_size_gen = 128
        start = traj_id // batch_size_gen * batch_size_gen
        end = start + batch_size_gen 
        if end > self.n_seq // self.n_seq_per_traj:
            end = self.n_seq // self.n_seq_per_traj
        print(f'generating {start}-{end-1}')
        with torch.no_grad():
            w0 = self._get_init_cond(traj_id, start, end).to(self.device)
            state, _ = self.navier_stokes_2d(w0, f=self.params_eq['f'].to(self.device), visc=self.params_eq['visc'],
                                             T=self.t_horizon * self.n_seq_per_traj, delta_t=self.dt, record_steps=self.n * self.n_seq_per_traj)
        state = state.permute(0, 4, 3, 1, 2)
        for i, ii in enumerate(range(start, end)):
            self.buffer_shelve[f'{ii}'] = {'data': state[i].cpu().numpy()}

#################
# SW-Sphere     #
#################


def build_s2_coord_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return np.meshgrid(phi_vert, theta_vert, indexing='ij')


class ShallowWaterDataset(AbstractDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_path = os.path.join(root, f"shallow_water_{'test' if self.group == 'test' else 'train'}")
        self.files_obj_buf = dict()
        self._load_trajectory(0, file_object_only=True)
        coords_list = []
        if self.group == 'test_hr':
            phi = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[1][0][:].ravel())
            theta = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[2][0][:].ravel())
        else:
            phi = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[1][0][:].ravel()[::2])
            theta = torch.tensor(self.files_obj_buf[0]['tasks/vorticity'].dims[2][0][:].ravel()[::2])

        spherical = get_mgrid_from_tensors([phi, theta])
        phi_vert = spherical[..., 0]
        theta_vert = spherical[..., 1]
        r = 1
        x = torch.cos(phi_vert) * torch.sin(theta_vert) * r
        y = torch.sin(phi_vert) * torch.sin(theta_vert) * r
        z = torch.cos(theta_vert) * r
        coords_list.append(torch.stack([x, y, z], dim=-1))

        self.coords_ang = get_mgrid_from_tensors([phi, theta])
        self.coords = torch.cat(coords_list, dim=-1).float()
        self.coord_dim = self.coords.shape[-1]

    def _load_trajectory(self, traj_id, file_object_only=False):
        if self.files_obj_buf.get(traj_id) is None:
            self.files_obj_buf[traj_id] = h5py.File(os.path.join(self.dataset_path, f'traj_{traj_id:04d}.h5'), mode='r')
        if file_object_only:
            return
        f = self.files_obj_buf[traj_id]
        if self.group == 'test_hr':
            return {'data': torch.stack([
                torch.from_numpy(f['tasks/height'][...]) * 3000.,
                torch.from_numpy(f['tasks/vorticity'][...] * 2),
                ], dim=0)}
        return {'data': torch.stack([
            torch.from_numpy(f['tasks/height'][:, ::2, ::2]) * 3000.,
            torch.from_numpy(f['tasks/vorticity'][:, ::2, ::2] * 2),
            ], dim=0)}
    
def extract_data(fp, variables):
    loaded_file = netCDFDataset(fp, 'r')
    data_dict = {}
    for var in variables:
        data_dict[var] = loaded_file.variables[var][:].data
    return data_dict

class SST(Dataset):
    var_names = ['thetao', 'daily_mean', 'daily_std']

    def __init__(self, data_dir, nt_cond, nt_pred, train, zones=range(1, 30)):
        super(SST, self).__init__()

        self.data_dir = data_dir
        self.nt_pred = nt_pred
        self.zones = list(zones)
        self.nt_cond = nt_cond
        self.zone_size = 64

        self.data = {}
        self.cst = {}
        self.climato = {}

        self.train = train

        self._normalize()

        self.first = 0 if self.train else int(0.8 * self.len_)

        self.coords = get_mgrid(self.zone_size, vmin=-1., vmax=1., dim=2)
        self.coord_dim = self.coords.shape[-1]

        # Retrieve length
        
        if self.train:
            self.len_ = int(0.8 * self.len_)
        else:
            self.len_ = self.len_ - int(0.8 * self.len_)

        self.len_ = int(self.len_ * 0.1)

        self.len_ = self.len_ - self.nt_pred - self.nt_cond - 1
        self._total_len = len(self.zones) * self.len_

    def _normalize(self):
        for zone in self.zones:
            zdata = extract_data(os.path.join(self.data_dir, f'data_{zone}.nc'), variables=self.var_names)
            self.len_ = len(zdata["thetao"])

            climate_mean, climiate_std = zdata['daily_mean'].reshape(-1, 1, 1), zdata['daily_std'].reshape(-1, 1, 1)
            zdata["thetao"] = (zdata["thetao"] - climate_mean) / climiate_std
            self.climato[zone] = (climate_mean, climiate_std)

            mean = zdata["thetao"].mean(axis=(1, 2)).reshape(-1, 1, 1)
            std = zdata["thetao"].std(axis=(1, 2)).reshape(-1, 1, 1)
            zdata["thetao"] = (zdata["thetao"] - mean) / std
            self.cst[zone] = (mean, std)
            self.data[zone] = zdata["thetao"]

    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        t = torch.arange(0, self.nt_pred + self.nt_cond, 1).float()
        file_id = self.zones[idx // self.len_]
        idx_id = (idx % self.len_ * 10) + self.nt_cond + 1 + self.first
        
        inputs = self.data[file_id][idx_id - self.nt_cond + 1: idx_id + 1].reshape(self.nt_cond, 1, self.zone_size,
                                                                              self.zone_size)
        target = self.data[file_id][idx_id + 1: idx_id + self.nt_pred + 1].reshape(self.nt_pred, 1, self.zone_size,
                                                                                  self.zone_size)
        inputs = torch.tensor(inputs, dtype=torch.float)
        target = torch.tensor(target, dtype=torch.float)

        return {
            'data': torch.cat([inputs, target], dim=0).permute(0, 2, 3, 1), 
            't': t, 
            'index': idx, 
            'coords': self.coords,
            'mu_clim': self.climato[file_id][0][idx_id + 1: idx_id + self.nt_pred + 1],
            'std_clim': self.climato[file_id][1][idx_id + 1: idx_id + self.nt_pred + 1],
            'mu_norm': self.cst[file_id][0][idx_id + 1: idx_id + self.nt_pred + 1],
            'std_norm': self.cst[file_id][1][idx_id + 1: idx_id + self.nt_pred + 1],
        }

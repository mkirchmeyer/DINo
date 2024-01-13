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

from torch.utils.data import DataLoader
from torch.nn import init
from torch import nn
import shelve
from data_pdes import WaveDataset, NavierStokesDataset, ShallowWaterDataset, SST
import math
import torch
from logging.handlers import RotatingFileHandler
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torchdiffeq import odeint


def process_config(input_dataset, path_results, device="gpu:0", mask_data=0.0, n_frames_train=10):
    if input_dataset == "wave":
        state_dim = 2
        code_dim = 50
        size = 64
        hidden_c = 256
        hidden_c_enc = 64
        n_layers = 3
        minibatch_size = 32
        dataset_tr_params = {
            "n_seq": 512, "n_seq_per_traj": 8, "t_horizon": 5, "dt": 0.25, "size": size, "group": "train",
            'n_frames_train': n_frames_train, "param": {"speed": 1/16, 'bc': 'periodic'}}
        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train_eval"
        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        buffer_file_tr = f"{path_results}/wave_train.shelve"
        buffer_shelve_tr = buffer_shelve_tr_eval = shelve.open(buffer_file_tr)
        buffer_file_ts = f"{path_results}/wave_test.shelve"
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        dataset_ts_params["n_seq"] = 32
        dataset_tr = WaveDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params)
        dataset_tr_eval = WaveDataset(buffer_shelve=buffer_shelve_tr_eval, **dataset_tr_eval_params)
        dataset_ts = WaveDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params)
        coord_dim = dataset_tr.coord_dim
    elif input_dataset == "navier_stokes":
        state_dim = 1
        code_dim = 100
        coord_dim = 2
        hidden_c = 512
        hidden_c_enc = 64
        n_layers = 3
        size = 64
        n_seq = 512
        t_horizon = 20
        minibatch_size = 32
        tt = torch.linspace(0, 1, size + 1)[0:-1]
        X, Y = torch.meshgrid(tt, tt)
        visc = 1e-3
        dataset_tr_params = {
            "device": "cuda:0", "n_seq": n_seq, "n_seq_per_traj": 2, "t_horizon": t_horizon, "dt": 1, "size": size,
            "group": "train", 'n_frames_train': n_frames_train,
            "param": {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": visc}
        }
        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train_eval"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test"
        dataset_ts_params["n_seq"] = 32

        buffer_file_tr = f"{path_results}/navier_1e-3_train.shelve"
        buffer_file_ts = f"{path_results}/navier_1e-3_test.shelve"

        buffer_shelve_tr = buffer_shelve_tr_eval = shelve.open(buffer_file_tr)
        buffer_shelve_ts = shelve.open(buffer_file_ts)
        dataset_tr = NavierStokesDataset(buffer_shelve=buffer_shelve_tr, **dataset_tr_params)
        dataset_tr_eval = NavierStokesDataset(buffer_shelve=buffer_shelve_tr_eval, **dataset_tr_eval_params)
        dataset_ts = NavierStokesDataset(buffer_shelve=buffer_shelve_ts, **dataset_ts_params)
    elif "shallow_water" in input_dataset:
        state_dim = 2
        coord_dim = 3
        code_dim = 200
        hidden_c = 800
        hidden_c_enc = 256
        n_layers = 6
        minibatch_size = 4
        size = (128, 64)
        n_seq = 64
        dataset_tr_params = {
            'dataset_name': 'shallow_water', 'root': f'{path_results}',  # Path to your generated data.
            "device": "cuda:0", 'buffer_shelve': None, "n_seq": n_seq, "n_seq_per_traj": 8, "t_horizon": 20, "dt": 1,
            "size": size, "group": "train", 'n_frames_train': n_frames_train
        }
        dataset_tr_eval_params = dict()
        dataset_tr_eval_params.update(dataset_tr_params)
        dataset_tr_eval_params["group"] = "train_eval"

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["group"] = "test" if not "hr" in input_dataset else "test_hr"
        dataset_ts_params["n_seq"] = 16

        dataset_tr = ShallowWaterDataset(**dataset_tr_params)
        dataset_tr_eval = ShallowWaterDataset(**dataset_tr_eval_params)
        dataset_ts = ShallowWaterDataset(**dataset_ts_params)
    elif input_dataset == "sst":
        state_dim = 1
        coord_dim = 2
        code_dim = 400
        hidden_c = 800
        hidden_c_enc = 256
        n_layers = 6
        minibatch_size = 32
        size = (64, 64)
        dataset_tr_params = {
            'data_dir': '/path/to/sst/dataset',
            'nt_cond': 4,
            'nt_pred': 6,
            'train': True, 
            'zones': range(17, 21),
        }

        dataset_ts_params = dict()
        dataset_ts_params.update(dataset_tr_params)
        dataset_ts_params["train"] = False
        dataset_ts_params["zones"] = range(17, 21)


        dataset_tr = SST(**dataset_tr_params)
        dataset_ts = SST(**dataset_ts_params)
        dataset_tr_eval = dataset_ts
        
        dataset_tr_params['n_seq'] = len(dataset_tr)
        dataset_ts_params['n_seq'] = len(dataset_ts)
        dataset_tr_eval_params = dataset_tr_params
    else:
        raise Exception(f"{input_dataset} does not exist")
    if isinstance(size, int):
        size = (size, size)
    n_mask = 1
    mask = generate_mask(size[0], size[1], device, mask_data, n_mask)
    mask_ts = mask

    
    if input_dataset == "shallow_water_hs":
        mask = generate_skipped_lat_lon_mask(dataset_tr.coords_ang, device).bool()
        mask_ts = generate_skipped_lat_lon_mask(dataset_ts.coords_ang, device, base_jump=1).bool()
    elif input_dataset == "shallow_water":
        mask = generate_skipped_lat_lon_mask(dataset_tr.coords_ang, device).bool()
        mask_ts = mask

    dataloader_tr = DataLoaderODE(dataset_tr, minibatch_size)
    dataloader_tr_eval = DataLoaderODE(dataset_tr_eval, minibatch_size, is_train=False)
    dataloader_ts = DataLoaderODE(dataset_ts, minibatch_size, is_train=False)
    return mask, mask_ts, size, state_dim, coord_dim, code_dim, hidden_c, hidden_c_enc, n_layers, \
           dataset_tr_params, dataset_tr_eval_params, dataset_ts_params, dataloader_tr, dataloader_tr_eval, dataloader_ts


def generate_skipped_lat_lon_mask(coords, device, base_jump=0):
    lons = coords[:, 0, 0].cpu().numpy()
    lats = coords[0, :, 1].cpu().numpy()
    n_lon = lons.size
    delta_dis_equator = 2 * np.pi * 1 / n_lon
    mask_list = []
    for lat in lats:
        delta_dis_lat = 2 * np.pi * np.sin(lat) / n_lon
        ratio = delta_dis_lat / delta_dis_equator
        n = int(np.ceil(np.log(ratio) / np.log(2/5)))
        mask = torch.zeros(n_lon)
        mask[::2 ** (n-1 + base_jump)] = 1
        mask_list.append(mask)

    mask = torch.stack(mask_list, dim=-1)
    return mask.to(device)


def generate_mask(h_size, w_size, device, mask_data=0, n_mask=1):
    mask_list = []
    for _ in range(n_mask):
        mask_list.append((torch.rand(h_size, w_size) >= mask_data)[None, :])
    mask = torch.cat(mask_list, dim=0).squeeze()
    return mask.to(device)


def eval_dino(dataloader, net_dyn, net_dec, device, method, criterion, mask_data, mask, state_dim, code_dim,
              coord_dim, n_frames_train=0, states_params=None, lr_adapt=0.0, dataset_params=None, n_steps=300,
              save_best=True):
    """
    In_t: loss within train horizon.
    Out_t: loss outside train horizon.
    In_s: loss within observation grid.
    Out_s: loss outside observation grid.
    loss: loss averaged across in_t/out_t and in_s/out_s
    loss_in_t: loss averaged across in_s/out_s for in_t.
    loss_in_t_in_s, loss_in_t_out_s: loss in_t + in_s / out_s
    """
    loss, loss_out_t, loss_in_t, loss_in_t_in_s, loss_in_t_out_s, loss_out_t_in_s, loss_out_t_out_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    gts, mos = [], []
    set_requires_grad(net_dec, False)
    set_requires_grad(net_dyn, False)
    for j, batch in enumerate(dataloader):
        ground_truth = batch['data'].to(device)
        t = batch['t'][0].to(device)
        index = batch['index'].to(device)
        model_input = batch['coords'].to(device)
        b_size, t_size, h_size, w_size, _ = ground_truth.shape
        if lr_adapt != 0.0:
            loss_min_test = 1e30
            states_params_out = nn.ParameterList([nn.Parameter(torch.zeros(1, code_dim * state_dim).to(device)) for _ in range(dataset_params["n_seq"])])
            optim_states_out = torch.optim.Adam(states_params_out, lr=lr_adapt)
            for i in range(n_steps):
                states_params_index = [states_params_out[d] for d in index]
                states_params_index = torch.stack(states_params_index, dim=1)
                states = states_params_index.permute(1, 0, 2).view(b_size, 1, state_dim, code_dim)
                model_input_exp = model_input.view(b_size, 1, h_size, w_size, 1, coord_dim)
                model_input_exp = model_input_exp.expand(b_size, 1, h_size, w_size, state_dim, coord_dim)
                model_output, _ = net_dec(model_input_exp, states)
                loss_l2 = criterion(model_output[:, :, mask, :], ground_truth[:, 0:1, mask, :])
                if loss_l2 < loss_min_test and save_best:
                    loss_min_test = loss_l2
                    best_states_params_index = states_params_index
                loss_opt_new = loss_l2

                loss_opt = loss_opt_new
                optim_states_out.zero_grad(True)
                loss_opt.backward()
                optim_states_out.step()
            if save_best:
                states_params_index = best_states_params_index
        with torch.no_grad():
            if lr_adapt == 0.0:
                states_params_index = [states_params[d] for d in index]
                states_params_index = torch.stack(states_params_index, dim=1)
            model_input_exp = model_input.view(b_size, 1, h_size, w_size, 1, coord_dim)
            model_input_exp = model_input_exp.expand(b_size, t_size, h_size, w_size, state_dim, coord_dim)
            codes = odeint(net_dyn, states_params_index[0], t, method=method)  # t x batch x dim
            codes = codes.permute(1, 0, 2).view(b_size, t_size, state_dim, code_dim)  # batch x t x dim
            model_output, _ = net_dec(model_input_exp, codes)
            if n_frames_train != 0:
                loss_in_t += criterion(model_output[:, :n_frames_train, :, :, :], ground_truth[:, :n_frames_train, :, :, :])
                loss += criterion(model_output, ground_truth)
            loss_out_t += criterion(model_output[:, n_frames_train:, :, :, :], ground_truth[:, n_frames_train:, :, :, :])
            if mask_data != 0.0:
                loss_in_t_in_s += criterion(model_output[:, :n_frames_train, mask, :], ground_truth[:, :n_frames_train, mask, :])
                loss_in_t_out_s += criterion(model_output[:, :n_frames_train, ~mask, :], ground_truth[:, :n_frames_train, ~mask, :])
                loss_out_t_in_s += criterion(model_output[:, n_frames_train:, mask, :], ground_truth[:, n_frames_train:, mask, :])
                loss_out_t_out_s += criterion(model_output[:, n_frames_train:, ~mask, :], ground_truth[:, n_frames_train:, ~mask, :])
            gts.append(ground_truth.cpu())
            mos.append(model_output.cpu())
    loss /= len(dataloader)
    loss_in_t /= len(dataloader)
    loss_out_t /= len(dataloader)
    loss_out_t_in_s /= len(dataloader)
    loss_out_t_out_s /= len(dataloader)
    loss_in_t_in_s /= len(dataloader)
    loss_in_t_out_s /= len(dataloader)
    set_requires_grad(net_dec, True)
    set_requires_grad(net_dyn, True)
    return loss, loss_in_t, loss_in_t_in_s, loss_in_t_out_s, loss_out_t, loss_out_t_in_s, loss_out_t_out_s, gts, mos

def eval_dino_cond(dataloader, net_dyn, net_dec, net_cond, device, method, criterion, mask_data, mask, state_dim, code_dim,
              coord_dim, n_frames_train=0, states_params=None, lr_adapt=0.0, input_dataset=None, n_steps=300, n_cond=4, is_test=True):
    loss, loss_out_t, loss_in_t = 0.0, 0.0, 0.0
    gts, mos, times, ss, pss, cs = [], [], [], [], [], []
    set_requires_grad(net_dec, False)
    set_requires_grad(net_dyn, False)
    set_requires_grad(net_cond, False)
    for j, batch in enumerate(dataloader):
        ground_truth = batch['data'].to(device)
        t = batch['t'][0][n_cond:].to(device)
        b_size, t_size, h_size, w_size, _ = ground_truth.shape
        index = batch['index'].to(device)
        model_input = batch['coords'].to(device)
        if lr_adapt != 0.0:
            states_params_out = nn.ParameterList([nn.Parameter(torch.zeros(n_cond + 1, code_dim * state_dim).to(device)) for _ in range(b_size)])
            optim_states_out = torch.optim.Adam(states_params_out, lr=lr_adapt)
            for i in range(n_steps):
                states_params_index = torch.stack(list(states_params_out), dim=1)  
                states = states_params_index.permute(1, 0, 2).view(b_size, n_cond + 1, state_dim, code_dim)
                model_input_exp = model_input.view(b_size, 1, h_size, w_size, 1, coord_dim)
                model_input_exp = model_input_exp.expand(b_size, n_cond + 1, h_size, w_size, state_dim, coord_dim)
                model_output, _ = net_dec(model_input_exp, states)
                loss_l2 = criterion(model_output[:, :, mask, :], ground_truth[:, :n_cond + 1, mask, :])
                loss_opt_new = loss_l2
                loss_opt = loss_opt_new
                optim_states_out.zero_grad(True)
                loss_opt.backward()
                optim_states_out.step()
        with torch.no_grad():
            if lr_adapt == 0.0:
                states_params_index = [states_params[d] for d in index]
                states_params_index = torch.stack(states_params_index, dim=1)
                states = states_params_index.permute(1, 0, 2).view(b_size, n_frames_train, state_dim, code_dim)
            model_input_exp = model_input.view(b_size, 1, h_size, w_size, 1, coord_dim)
            model_input_exp = model_input_exp.expand(b_size, t_size-n_cond, h_size, w_size, state_dim, coord_dim)
            extra_state = net_cond(states_params_index[:n_cond].permute(1, 0, 2).detach().clone())
            augmented_state = torch.cat([extra_state, states_params_index[n_cond].detach().clone()], dim=-1)

            codes = odeint(net_dyn, augmented_state, t, method=method)  # t x batch x dim
            codes = codes[:, :, code_dim * state_dim:].permute(1, 0, 2).view(b_size, t.numel(), state_dim, code_dim)  # batch x t x dim

            model_output, _ = net_dec(model_input_exp, codes)

            ground_truth_ = ground_truth[:, n_cond:n_frames_train, :, :, :]
            model_output_ = model_output

            if input_dataset == "sst":
                mu_norm, std_norm = batch['mu_norm'].to(device).unsqueeze(-1), batch['std_norm'].to(device).unsqueeze(-1)

                model_output_ = (model_output_ * std_norm) + mu_norm
                ground_truth_ = (ground_truth_ * std_norm) + mu_norm

                # Original space for MSE
                mu_clim, std_clim = batch['mu_clim'].to(device).unsqueeze(-1), batch['std_clim'].to(device).unsqueeze(-1)
                model_output_ = (model_output_ * std_clim) + mu_clim
                ground_truth_ = (ground_truth_ * std_clim) + mu_clim

            if n_frames_train != 0:
                loss_in_t += criterion(model_output_[:, :n_frames_train-n_cond, :, :, :], ground_truth_)
                loss += criterion(model_output_[:, :n_frames_train-n_cond, :, :, :], ground_truth_)
            if mask_data != 0.0:
                loss_in_t_in_s += criterion(model_output_[:, :n_frames_train, mask, :], ground_truth[:, :n_frames_train, mask, :])
                loss_in_t_out_s += criterion(model_output_[:, :n_frames_train, ~mask, :], ground_truth[:, :n_frames_train, ~mask, :])
            gts.append(ground_truth.cpu())
            mos.append(model_output.cpu())
            pss.append(torch.zeros(1))
            times.append(t.cpu())
            ss.append(states.cpu())
            cs.append(codes.cpu())
        print(j)
        if not is_test:
            break
    loss /= (j+1)
    loss_in_t /= (j+1)

    set_requires_grad(net_dec, True)
    set_requires_grad(net_dyn, True)
    set_requires_grad(net_cond, True)
    
    return loss, loss_in_t, gts, mos, times, ss, pss, cs


def scheduling(_int, _f, true_codes, t, epsilon, method='rk4'):
    if epsilon < 1e-3:
        epsilon = 0
    if epsilon == 0:
        codes = _int(_f, y0=true_codes[0], t=t, method=method)
    else:
        eval_points = np.random.random(len(t)) < epsilon
        eval_points[-1] = False
        eval_points = eval_points[1:]
    
        start_i, end_i = 0, None
        codes = []
        for i, eval_point in enumerate(eval_points):
            if eval_point == True:
                end_i = i + 1
                t_seg = t[start_i:end_i + 1]
                res_seg = _int(_f, y0=true_codes[start_i], t=t_seg, method=method)
                
                if len(codes) == 0:
                    codes.append(res_seg)
                else:
                    codes.append(res_seg[1:])
                start_i = end_i
        t_seg = t[start_i:]
        res_seg = _int(_f, y0=true_codes[start_i], t=t_seg, method=method)
        if len(codes) == 0:
            codes.append(res_seg)
        else:
            codes.append(res_seg[1:])
        codes = torch.cat(codes, dim=0)
    return codes


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('Bilinear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'default':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if init_type != 'default' and hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


def create_logger(folder, outfile):
    try:
        os.makedirs(folder)
        print(f"Directory {folder} created")
    except FileExistsError:
        print(f"Directory {folder} already exists replacing files in this notebook")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_handler = RotatingFileHandler(outfile, "w")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)
    return logger


def DataLoaderODE(dataset, minibatch_size, is_train=True):
    dataloader_params = {
        'dataset': dataset,
        'batch_size': minibatch_size,
        'shuffle': is_train,
        'num_workers': 0,  # for main thread
        'pin_memory': True,
        'drop_last': False
    }
    return DataLoader(**dataloader_params)


def write_image(batch_gt, batch_pred, state_idx, path, cmap='plasma', divider=1):
    """
    Print reference trajectory (1st line) and predicted trajectory (2nd line).
    Skip every N frames (N=divider)
    """
    batch_gt = torch.permute(batch_gt, (1, 0, 2, 3, 4))
    batch_pred = torch.permute(batch_pred, (1, 0, 2, 3, 4))
    seq_len, batch_size, height, width, state_c = batch_gt.shape  # [8, 20, 64, 64, 2]
    t_horizon = math.ceil(seq_len / divider)
    fig = plt.figure(figsize=(t_horizon, batch_size * 2.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(batch_size * 2, t_horizon),  # creates 2x2 grid of axes
                     axes_pad=0.05)  # pad between axes in inch.
    for traj in range(batch_size):
        vmax = torch.max(batch_gt[:, traj, :, :, :]).cpu().numpy()
        vmin = torch.min(batch_gt[:, traj, :, :, :]).cpu().numpy()
        for t in range(t_horizon):
            # Iterating over the grid returns the Axes.
            grid[2 * traj * t_horizon + t].imshow(batch_gt[divider * t, traj, :, :, state_idx].cpu().numpy(), vmax=vmax, vmin=vmin, cmap=cmap, interpolation='none')
            if t - 4 >= 0:
                grid[(2 * traj + 1) * t_horizon + t].imshow(batch_pred[divider * t - 4, traj, :, :, state_idx].cpu().numpy(), vmax=vmax, vmin=vmin, cmap=cmap, interpolation='none')
            grid[2 * traj * t_horizon + t].set_axis_off()
            grid[(2 * traj + 1) * t_horizon + t].set_axis_off()

    plt.savefig(os.path.join(path), dpi=72, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf


def set_rdm_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

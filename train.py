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

import matplotlib
from ode_model import Decoder, Derivative
from network import MLP, SetEncoder
from datetime import datetime
from torchdiffeq import odeint
import getopt
import sys
import os
import logging
from torch import nn
import matplotlib.pyplot as plt
import torch
from utils import process_config, count_parameters, set_rdm_seed, create_logger, scheduling, write_image, eval_dino, eval_dino_cond
from matplotlib import rcParams

logging.getLogger('numba').setLevel(logging.CRITICAL)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger('PIL').setLevel(logging.WARNING)
matplotlib.pyplot.set_loglevel("critical")

log_every = 8
input_dataset = "navier"
gpu = 0
gpu_id = 0
home_folder = "./results"
lr = 1e-2
lr_adapt = 1e-2
seed = 1
options = {}
opts, args = getopt.getopt(sys.argv[1:], "c:d:f:g:r:w:")
subsampling_rate = 1.0
checkpoint_path = None  # warm start from a model in this path
n_cond = 0
for opt, arg in opts:
    if opt == "-c":
        checkpoint_path = arg
    if opt == "-d":
        input_dataset = arg
    if opt == "-f":
        home_folder = arg
    if opt == "-g":
        gpu = int(arg)
    if opt == "-r":
        subsampling_rate = float(arg)
    if opt == "-w":
        n_cond = int(arg)

mask_data = 1. - subsampling_rate
now = datetime.now()
ts = now.strftime("%Y%m%d_%H%M%S")
cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')

path_results = os.path.join(home_folder, input_dataset)
path_checkpoint = os.path.join(path_results, ts)
logger = create_logger(path_checkpoint, os.path.join(path_checkpoint, "log"))
os.makedirs(path_checkpoint, exist_ok=True)
init_type = "default"
set_rdm_seed(seed)

# Config
first = 4
n_frames_train = 10
mask, mask_ts, size, state_dim, coord_dim, code_dim, hidden_c, hidden_c_enc, n_layers, dataset_tr_params, \
dataset_tr_eval_params, dataset_ts_params, dataloader_tr, dataloader_tr_eval, dataloader_ts = \
    process_config(input_dataset, path_results, mask_data=mask_data, device=device, n_frames_train=n_frames_train)
epsilon = epsilon_t = 0.99
eval_every = 100
n_epochs = 120000
method = "rk4" if n_cond == 0 else "euler" 

if input_dataset == "wave" or input_dataset == "shallow_water":
    n_steps = 500
else:
    n_steps = 300

if checkpoint_path is None:  # Start from scratch
    # Decoder
    net_dec_params = {
        'state_c': state_dim, 
        'code_c': code_dim, 
        'hidden_c': hidden_c_enc, 
        'n_layers': n_layers, 
        'coord_dim': coord_dim
    }
    # Forecaster
    net_dyn_params = {
        'state_c': state_dim, 
        'hidden_c': hidden_c, 
        'code_c': code_dim if n_cond == 0 else code_dim * 2
    }
    net_dec = Decoder(**net_dec_params)
    net_dyn = Derivative(**net_dyn_params)
    if n_cond > 0:
        net_cond_params = {
            'code_size': code_dim * state_dim,
            'n_cond': n_cond,
            'hidden_size': 1024
        }
        net_cond = SetEncoder(**net_cond_params)
    states_params = nn.ParameterList([nn.Parameter(torch.zeros(n_frames_train, code_dim * state_dim).to(device)) for _ in range(dataset_tr_eval_params["n_seq"])])

    print(dict(net_dec.named_parameters()).keys())
    print(dict(net_dyn.named_parameters()).keys())

    net_dec = net_dec.to(device)
    net_dyn = net_dyn.to(device)
    if n_cond > 0:
        net_cond = net_cond.to(device)
else:  # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}')
    logger.info(f"N_ones: {torch.sum(mask_ts)}")
    logger.info(f"Missingness: {100. * (1 - torch.sum(mask_ts) / (size[0] * size[1]))}%")
    net_dec_params = checkpoint["net_dec_params"]
    state_dim = net_dec_params['state_c']
    code_dim = net_dec_params['code_c']
    net_dec = Decoder(**net_dec_params)
    net_dec_dict = net_dec.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['dec_state_dict'].items() if k in net_dec_dict}
    net_dec_dict.update(pretrained_dict)
    net_dec.load_state_dict(net_dec_dict)
    print(dict(net_dec.named_parameters()).keys())

    net_dyn_params = checkpoint["net_dyn_params"]
    net_dyn = Derivative(**net_dyn_params)
    net_dyn_dict = net_dyn.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['dyn_state_dict'].items() if k in net_dyn_dict}
    net_dyn_dict.update(pretrained_dict)
    net_dyn.load_state_dict(net_dyn_dict)
    print(dict(net_dyn.named_parameters()).keys())

    if n_cond > 0:
        net_cond = SetEncoder(code_dim, n_cond, 1024)
        net_cond_dict = net_cond.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['cond_state_dict'].items() if k in net_dyn_dict}
        net_cond_dict.update(pretrained_dict)
        net_cond.load_state_dict(net_cond_dict)
        print(dict(net_cond.named_parameters()).keys())

    states_params = checkpoint["states_params"]
    net_dec = net_dec.to(device)
    net_dyn = net_dyn.to(device)
    if n_cond > 0:
        net_cond = net_cond.to(device)

criterion = nn.MSELoss()

optim_net_dec = torch.optim.Adam([{'params': net_dec.parameters(), 'lr': lr}])
optim_net_dyn = torch.optim.Adam([{'params': net_dyn.parameters(), 'lr': lr / 10}])
optim_states = torch.optim.Adam([{'params': states_params, 'lr': lr / 10}])
if n_cond:
    optim_net_cond = torch.optim.Adam([{'params': net_cond.parameters(), 'lr': lr / 10}])

# Logs
logger.info(f"run_id: {ts}")
if cuda:
    logger.info(f"gpu_id: {gpu_id}")
logger.info(f"seed: {seed}")
logger.info(f"dataset: {input_dataset}")
logger.info(f"method: {method}")
logger.info(f"code_c: {code_dim}")
logger.info(f"lr: {lr}")
logger.info(f"n_params forecaster: {count_parameters(net_dec) + count_parameters(net_dyn)}")
logger.info(f"coord_dim: {coord_dim}")
logger.info(f"n_frames_train: {n_frames_train}")
logger.info(f"subsampling_rate: {subsampling_rate*100}%")
if n_cond > 0:
    logger.info(f"n_cond: {n_cond}")

# Train
loss_tr_min, loss_ts_min, loss_relative_min = float('inf'), float('inf'), float('inf')
for epoch in range(n_epochs):
    # Update Decoder and Dynamics
    if n_cond == 0:
        if epoch != 0:
            optim_net_dec.step()
            optim_net_dec.zero_grad()

            optim_net_dyn.step()
            optim_net_dyn.zero_grad()

        for i, batch in enumerate(dataloader_tr):
            ground_truth = batch['data'].to(device)
            model_input = batch['coords'].to(device) 
            t = batch['t'][0].to(device)
            index = batch['index'].to(device)
            b_size, t_size, h_size, w_size, _ = ground_truth.shape
            if epoch == 0 and i == 0:
                # Display info on grid subsampling
                logger.info(f"N_ones in mask: {torch.sum(mask)}")
                logger.info(f"Missingness ratio: {100. * (1 - torch.sum(mask) / (w_size * h_size))}%")
                plt.imshow(mask.cpu().numpy(), interpolation='none')
                plt.savefig(os.path.join(path_checkpoint, f"mask.png"), dpi=72, bbox_inches='tight', pad_inches=0)
                
                plt.imshow(mask_ts.cpu().numpy(), interpolation='none')
                plt.savefig(os.path.join(path_checkpoint, f"mask_ts.png"), dpi=72, bbox_inches='tight', pad_inches=0)
                logger.info(f"ground_truth: {list(ground_truth.size())}")
                logger.info(f"t: {t[0]}")
                logger.info(f"index: {index}")

            # Update latent states
            states_params_index = torch.stack([states_params[d] for d in index], dim=1)
            states = states_params_index.permute(1, 0, 2).view(b_size, t_size, state_dim, code_dim)
            model_input_exp = model_input.view(b_size, 1, h_size, w_size, 1, coord_dim).expand(b_size, t_size, h_size, w_size, state_dim, coord_dim)
            model_output, _ = net_dec(model_input_exp, states)
            loss_l2 = criterion(model_output[:, :, mask, :], ground_truth[:, :, mask, :])
            optim_states.zero_grad(True)
            loss_l2.backward()
            optim_states.step()

            # Cumulate gradient of dynamics
            codes = scheduling(odeint, net_dyn, states_params_index.detach().clone(), t, epsilon_t, method=method)
            loss_l2_states = criterion(codes, states_params_index.detach().clone())
            loss_l2_states.backward()
            if (epoch * len(dataloader_tr) + i) % log_every == 0:
                logger.info("Dataset %s, Runid %s, Epoch [%d/%d] MSE Auto-dec %0.3e, MSE Dyn %0.3e, epsilon %0.3e" % (
                    input_dataset, ts, epoch, i, loss_l2, loss_l2_states, epsilon_t))

            if (epoch * len(dataloader_tr) + i + 1) % eval_every == 0:
                epsilon_t *= epsilon
                print("Evaluating train...")
                loss_tr, loss_tr_in_t, loss_tr_in_t_in_s, loss_tr_in_t_out_s, loss_tr_out_t, loss_tr_out_t_in_s, \
                loss_tr_out_t_out_s, gts, mos = eval_dino(dataloader_tr_eval, net_dyn, net_dec, device,
                method, criterion, mask_data, mask_ts, state_dim, code_dim, coord_dim, n_frames_train, states_params, n_steps=n_steps)
                optimize_tr = loss_tr
                if loss_tr_min > optimize_tr:
                    logger.info(f"Checkpoint created: min tr loss was {loss_tr_min}, new is {optimize_tr}")
                    for j, (ground_truth, model_output) in enumerate(zip(gts, mos)):
                        if j in [0]:
                            for state_idx in range(state_dim):
                                write_image(ground_truth[:first], model_output[:first], state_idx, os.path.join(path_checkpoint, f"img_tr_state{state_idx}.pdf"))
                    loss_tr_min = optimize_tr
                    torch.save({
                        "epoch": epoch,
                        "dec_state_dict": net_dec.state_dict(),
                        "dyn_state_dict": net_dyn.state_dict(),
                        "states_params": states_params,
                        "loss_in_test": loss_tr_min,
                        "net_dec_params": net_dec_params,
                        "net_dyn_params": net_dyn_params,
                        "dataset_tr_params": dataset_tr_params
                        }, os.path.join(path_checkpoint, f"model_tr.pt"))

                # Out-of-domain evaluation
                print("Evaluating test...")
                loss_ts, loss_ts_in_t, loss_ts_in_t_in_s, loss_ts_in_t_out_s, loss_ts_out_t, loss_ts_out_t_in_s, \
                loss_ts_out_t_out_s, gts, mos = eval_dino(dataloader_ts, net_dyn, net_dec, device, method,
                criterion, mask_data, mask_ts, state_dim, code_dim, coord_dim, n_frames_train, states_params, lr, dataset_ts_params, n_steps=n_steps)
                optimize_ts = loss_ts
                if loss_ts_min > optimize_ts:
                    logger.info(f"Checkpoint created: min ts loss was {loss_ts_min}, new is {optimize_ts}")
                    for j, (ground_truth, model_output) in enumerate(zip(gts, mos)):
                        if j in [0]:
                            for state_idx in range(state_dim):
                                write_image(ground_truth[:first], model_output[:first], state_idx, os.path.join(path_checkpoint, f"img_ts_state{state_idx}.pdf"))
                    loss_ts_min = optimize_ts
                    torch.save({
                        "epoch": epoch,
                        "dec_state_dict": net_dec.state_dict(),
                        "dyn_state_dict": net_dyn.state_dict(),
                        "optim_net_dec": optim_net_dec.state_dict(),
                        "optim_net_dyn": optim_net_dyn.state_dict(),
                        "optim_states": optim_states.state_dict(),
                        "states_params": states_params,
                        "loss_out_test": loss_ts_min,
                        "net_dec_params": net_dec_params,
                        "net_dyn_params": net_dyn_params,
                        "dataset_tr_params": dataset_tr_params
                        }, os.path.join(path_checkpoint, f"model_ts.pt"))
                logger.info("Dataset %s, Runid %s, Epoch %d, Iter %d, Loss_tr: %.3e In-t: %.3e In-s: %.3e Out-s: %.3e "
                            "Out-t: %.3e In-s: %.3e Out-s: %.3e, Loss_ts: %.3e In-t: %.3e In-s: %.3e Out-s: %.3e Out-t: %.3e "
                            "In-s: %.3e Out-s: %.3e" % (input_dataset, ts, epoch + 1, i + 1, loss_tr, loss_tr_in_t,
                            loss_tr_in_t_in_s, loss_tr_in_t_out_s, loss_tr_out_t, loss_tr_out_t_in_s, loss_tr_out_t_out_s,
                            loss_ts, loss_ts_in_t, loss_ts_in_t_in_s, loss_ts_in_t_out_s, loss_ts_out_t, loss_ts_out_t_in_s, loss_ts_out_t_out_s))
                logger.info("========")

    else:
        for i, batch in enumerate(dataloader_tr):
            ground_truth = batch['data'].to(device)
            model_input = batch['coords'].to(device) 
            t = batch['t'][0][n_cond:].to(device)
            index = batch['index'].to(device)
            b_size, t_size, h_size, w_size, _ = ground_truth.shape
            if epoch == 0 and i == 0:
                logger.info(f"N_ones: {torch.sum(mask)}")
                logger.info(f"Missingness: {100. * (1 - torch.sum(mask) / (w_size * h_size))}%")
                logger.info(f"ground_truth: {list(ground_truth.size())}")
                logger.info(f"t: {t[0]}")
                logger.info(f"index: {index}")

            # Update states
            states_params_index = torch.stack([states_params[d] for d in index], dim=1)
            states = states_params_index.permute(1, 0, 2).view(b_size, t_size, state_dim, code_dim)
            model_input_exp = model_input.view(b_size, 1, h_size, w_size, 1, coord_dim).expand(b_size, t_size, h_size, w_size, state_dim, coord_dim)
            model_output, _ = net_dec(model_input_exp, states)
            loss_l2 = criterion(model_output[:, :, mask, :], ground_truth[:, :, mask, :])
            loss_opt = loss_l2

            optim_states.zero_grad(True)
            loss_opt.backward()
            optim_states.step()

            if (epoch * len(dataloader_tr) + i + 1) % 4 == 0:
                optim_net_dec.step()
                optim_net_dec.zero_grad()

            # Update Dynamics
            extra_states = []
            for jjj in range(n_frames_train - n_cond):
                extra_states.append(net_cond(states_params_index[jjj:jjj+n_cond].permute(1, 0, 2).detach().clone()))
            
            extra_states = torch.stack(extra_states, dim=0)
            augmented_states = torch.cat([extra_states, states_params_index[n_cond:].detach().clone()], dim=-1)

            codes = scheduling(odeint, net_dyn, augmented_states, t, epsilon_t, method=method)
            loss_l2_states = criterion(codes[:, :, code_dim * state_dim:], states_params_index[n_cond:].detach().clone())
            loss_opt_states = loss_l2_states
            
            loss_opt_states.backward()
            optim_net_dyn.step()
            optim_net_cond.step()
            optim_net_dyn.zero_grad()
            optim_net_cond.zero_grad()

            model_output_ = model_output.detach()[:, n_cond:]
            ground_truth_ = ground_truth[:, n_cond:]
            
            if input_dataset == 'sst':
                mu_norm, std_norm = batch['mu_norm'].to(device).unsqueeze(-1), batch['std_norm'].to(device).unsqueeze(-1)

                model_output_ = (model_output_ * std_norm) + mu_norm
                ground_truth_ = (ground_truth_ * std_norm) + mu_norm

                # Original space for MSE
                mu_clim, std_clim = batch['mu_clim'].to(device).unsqueeze(-1), batch['std_clim'].to(device).unsqueeze(-1)
                model_output_ = (model_output_ * std_clim) + mu_clim
                ground_truth_ = (ground_truth_ * std_clim) + mu_clim

            loss_l2_ = criterion(model_output_[:, :, mask, :], ground_truth_[:, :, mask, :])
            
            if (epoch * len(dataloader_tr) + i) % log_every == 0:
                logger.info("Dataset %s, Runid %s, Epoch [%d/%d] MSE Auto-dec %0.3e, MSE Dyn %0.3e, epsilon %0.3e" % (
                    input_dataset, ts, epoch, i, loss_l2_, loss_l2_states, epsilon_t))

            if (epoch * len(dataloader_tr) + i + 1) % eval_every == 0:
                epsilon_t *= epsilon

            if (epoch * len(dataloader_tr) + i + 1) % (eval_every * 5) == 0:
                print("Evaluating train...")
                loss_tr, loss_tr_in_t, gts, mos, times, ss, pss, cs = eval_dino_cond(dataloader_tr, net_dyn, net_dec, net_cond,
                    device, method, criterion, mask_data, mask, state_dim, code_dim, coord_dim, n_frames_train,
                    states_params, lr_adapt, input_dataset=input_dataset, is_test=False)
                optimize_tr = loss_tr
                if loss_tr_min > optimize_tr:
                    logger.info(f"Checkpoint created: min tr loss was {loss_tr_min}, new is {optimize_tr}")
                    for j, (ground_truth, model_output, codes, states, t) in enumerate(zip(gts, mos, cs, ss, times)):
                        if j in [0]:
                            for state_idx in range(state_dim):
                                write_image(ground_truth[:first], model_output[:first], state_idx,
                                            os.path.join(path_checkpoint, f"img_tr_state{state_idx}.pdf"))
                    loss_tr_min = optimize_tr
                    torch.save({
                        "epoch": epoch,
                        "dec_state_dict": net_dec.state_dict(),
                        "dyn_state_dict": net_dyn.state_dict(),
                        "cond_state_dict": net_cond.state_dict(),
                        "optim_net_dec": optim_net_dec.state_dict(),
                        "optim_net_dyn": optim_net_dyn.state_dict(),
                        "optim_net_cond": optim_net_cond.state_dict(),
                        "optim_states": optim_states.state_dict(),
                        "states_params": states_params,
                        "loss_out_test": loss_ts_min,
                        "net_dec_params": net_dec_params,
                        "net_dyn_params": net_dyn_params,
                        "net_cond_params": net_cond_params,
                        "epsilon_t": epsilon_t,
                        "dataset_tr_params": dataset_tr_params
                    }, os.path.join(path_checkpoint, f"model_tr.pt"))

                print("Evaluating test...")
                loss_ts, loss_ts_in_t, gts, mos, times, ss, pss, cs = eval_dino_cond(dataloader_ts, net_dyn, net_dec, net_cond, device, method,
                criterion, mask_data, mask_ts, state_dim, code_dim, coord_dim, n_frames_train, states_params, lr_adapt, input_dataset=input_dataset, is_test=True)
                optimize_ts = loss_ts
                if loss_ts_min > optimize_ts:
                    logger.info(f"Checkpoint created: min ts loss was {loss_ts_min}, new is {optimize_ts}")
                    for j, (ground_truth, model_output, codes, states, t) in enumerate(zip(gts, mos, cs, ss, times)):
                        if j in [0]:
                            for state_idx in range(state_dim):
                                write_image(ground_truth[:first], model_output[:first], state_idx, os.path.join(path_checkpoint, f"img_ts_state{state_idx}.pdf"), cmap="seismic")
                    loss_ts_min = optimize_ts
                    torch.save({
                        "epoch": epoch,
                        "dec_state_dict": net_dec.state_dict(),
                        "dyn_state_dict": net_dyn.state_dict(),
                        "cond_state_dict": net_cond.state_dict(),
                        "optim_net_dec": optim_net_dec.state_dict(),
                        'optim_net_dyn': optim_net_dyn.state_dict(),
                        'optim_net_cond': optim_net_cond.state_dict(),
                        'optim_states': optim_states.state_dict(),
                        "states_params": states_params,
                        "loss_out_test": loss_ts_min,
                        "net_dec_params": net_dec_params,
                        "net_dyn_params": net_dyn_params,
                        "net_cond_params": net_cond_params,
                        "epsilon_t": epsilon_t,
                        "dataset_tr_params": dataset_tr_params
                        }, os.path.join(path_checkpoint, f"model_ts.pt"))

                logger.info("Dataset %s, Runid %s, Epoch %d, Iter %d, Loss_ts: %.3e In-t: %.3e" % (input_dataset, ts, epoch + 1, i + 1,
                            loss_ts, loss_ts_in_t))
                logger.info("========")

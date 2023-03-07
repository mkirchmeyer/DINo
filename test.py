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
from network import SetEncoder
from datetime import datetime
import getopt
import sys
import os
import logging
from torch import nn
import matplotlib.pyplot as plt
import torch
from utils import process_config, set_rdm_seed, create_logger, write_image, eval_dino, eval_dino_cond

logging.getLogger('numba').setLevel(logging.CRITICAL)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger('PIL').setLevel(logging.WARNING)
matplotlib.pyplot.set_loglevel("critical")

input_dataset = "navier"
gpu = 0
gpu_id = 0
home_folder = "./results"
lr_adapt = 1e-2
seed = 1
options = {}
path_model = ""
n_steps = 300
method = "rk4"
subsampling_rate = 1.0
opts, args = getopt.getopt(sys.argv[1:], "d:f:g:p:r:s:")
for opt, arg in opts:
    if opt == "-d":
        input_dataset = arg
    if opt == "-f":
        home_folder = arg
    if opt == "-g":
        gpu = int(arg)
    if opt == "-p":
        path_model = arg
    if opt == "-r":
        subsampling_rate = float(arg)
    if opt == "-s":
        seed = int(arg)

if input_dataset == "wave" or input_dataset == "shallow_water":
    n_steps = 500
else:
    n_steps = 300

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
set_rdm_seed(seed)

# Config
first = 4
n_frames_train = 10
_, mask_ts, size, _, coord_dim, _, hidden_c, hidden_c_enc, n_layers, _, dataset_tr_eval_params, dataset_ts_params, _, _, \
dataloader_ts = process_config(input_dataset, path_results, mask_data=mask_data, device=device, n_frames_train=n_frames_train)

# Load checkpoint
checkpoint = torch.load(os.path.join(home_folder, input_dataset, path_model, 'model_ts.pt'), map_location=f'cuda:{gpu_id}')
logger.info(f"N_ones: {torch.sum(mask_ts)}")
logger.info(f"Missingness: {100. * (1 - torch.sum(mask_ts) / (size[0] * size[1]))}%")
plt.imshow(mask_ts.cpu().numpy(), interpolation='none')
plt.savefig(os.path.join(path_checkpoint, f"mask.png"), dpi=72, bbox_inches='tight', pad_inches=0)
plt.close()

is_markovian = "net_cond_params" not in checkpoint

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

if not is_markovian:
    net_cond_params = checkpoint["net_cond_params"]
    net_cond = SetEncoder(**net_cond_params)
    net_cond_dict = net_cond.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['cond_state_dict'].items() if k in net_dyn_dict}
    net_cond_dict.update(pretrained_dict)
    net_cond.load_state_dict(net_cond_dict)
    print(dict(net_cond.named_parameters()).keys())

net_dec = net_dec.to(device)
net_dyn = net_dyn.to(device)
if not is_markovian:
    net_cond = net_cond.to(device)
criterion = nn.MSELoss()

# Logs
logger.info(f"run_id: {ts}")

print("Evaluating test...")
if is_markovian:
    loss_ts, loss_ts_in_t, loss_ts_in_t_in_s, loss_ts_in_t_out_s, loss_ts_out_t, loss_ts_out_t_in_s, loss_ts_out_t_out_s, \
    gts, mos = eval_dino(
        dataloader_ts, net_dyn, net_dec, device, method, criterion, mask_data, mask_ts, state_dim, code_dim, coord_dim,
        n_frames_train=n_frames_train, lr_adapt=lr_adapt, dataset_params=dataset_ts_params, n_steps=n_steps, save_best=True)
    for j, (ground_truth, model_output) in enumerate(zip(gts, mos)):
        if j in [0]:
            for state_idx in range(state_dim):
                write_image(ground_truth[:first], model_output[:first], state_idx,
                            os.path.join(path_checkpoint, f"img_ts_state{state_idx}.pdf"))
    logger.info("Dataset %s, Runid %s, Loss_ts: %.3e In-t: %.3e In-s: %.3e Out-s: %.3e Out-t: %.3e In-s: %.3e Out-s: %.3e" % (
        input_dataset, ts, loss_ts, loss_ts_in_t, loss_ts_in_t_in_s, loss_ts_in_t_out_s, loss_ts_out_t, loss_ts_out_t_in_s, loss_ts_out_t_out_s))
    logger.info("========")
else:
    loss_ts, loss_ts_in_t, gts, mos, times, ss, pss, cs = eval_dino_cond(dataloader_ts, net_dyn, net_dec, net_cond, device, method,
                criterion, mask_data, mask_ts, state_dim, code_dim, coord_dim, n_frames_train, lr_adapt=lr_adapt, input_dataset=input_dataset, is_test=True)
    for j, (ground_truth, model_output, codes, states, t) in enumerate(zip(gts, mos, cs, ss, times)):
        if j in [0]:
            for state_idx in range(state_dim):
                write_image(ground_truth[:first], model_output[:first], state_idx, os.path.join(path_checkpoint, f"img_ts_state{state_idx}.pdf"), cmap="seismic")
    logger.info("Dataset %s, Runid %s, Loss_ts: %.3e In-t: %.3e" % (input_dataset, ts, loss_ts, loss_ts_in_t))
    logger.info("========")
import os, sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), '../..')))
import torch as tr
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import torch.optim as opt
from bnn_src.imp_bnn import create_generator_nn, ImplicitBNNs
from bnn_src import data_loaders
# from utils import data_utils, test
from stochman import nnj
from bnn_src.models import MMgenerator, BayesianMLP, BayesianLeNet, BayesianWideResNet, CorreMMGenerator
from tqdm import tqdm, trange
from bnn_src.metrics import get_calib
import yaml
import numpy as np
import argparse
import pickle
import math


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='cifar_gen_architecture')  #Default architecture is for CIFAR10
parser.add_argument('--old_model',type=bool, default=False)
parser.add_argument('--model_name', type=str)

args = parser.parse_args()

data_path = 'data/'
path = 'saved_models/dlb_CIFAR100_resnet/'
device = 'cuda:0' if tr.cuda.is_available() else 'cpu'

#data_loading

train_loader, val_loader, test_loader = data_loaders.get_cifar100_loaders(data_path, batch_size=128,
                                                                              train_batch_size=128, download=True)
targets = tr.cat([y for x, y in test_loader], dim=0)
num_classes = 100


#model_loading

model_name = args.model_name
load_dict = tr.load(path + model_name)

#model_definition
with open(f'gen_arch_config/{args.config}.yaml', 'r') as f:
	config = yaml.full_load(f)
	input_f, output_features, hidden_units, n_hid = config
generator = CorreMMGenerator(input_features=input_f, output_features=output_features, hidden_units=hidden_units,
                                         n_hidden=n_hid,activ_func=nnj.ELU)
generator.to(device)
bnn_clf = BayesianWideResNet(generator, 28, 10, 100)
bnn_clf.to(device)
final_out = sum(math.prod(v) for v in output_features.values())
base_dist = dist.Normal(loc=tr.zeros(bnn_clf.noise_dim, device=device), scale=tr.ones(bnn_clf.noise_dim, device=device))
prior_dist = dist.Normal(loc=tr.zeros(final_out, device=device), scale=tr.ones(final_out, device=device))
if args.old_model:
        imp_vi_cls = ImplicitBNNs(bnn_clf,prior_dist,base_dist,small_noise=0.051)
        imp_vi_cls.bnn.load_state_dict(tr.load(load_dict))
else:   #newer models work this way
        imp_vi_cls = ImplicitBNNs(bnn_clf,prior_dist,base_dist,small_noise=load_dict['small_noise'])
        imp_vi_cls.bnn.load_state_dict(load_dict['model_weights'])

del load_dict # do this to save memory? 

#evaluation
print('Evaluation Begins...')
imp_vi_cls.bnn.eval()
pred = imp_vi_cls.predict(test_loader, n_samples=20).cpu()
for _ in range(4):
        pred+=imp_vi_cls.predict(test_loader, n_samples=20).cpu()
#breakpoint()
pred = pred/5
acc_val = np.mean(np.argmax(pred.numpy(), 1) == targets.numpy()) * 100
mmc_val = pred.numpy().max(-1).mean() * 100

print(f'acccuracy on test set {acc_val}, mmc on test set {mmc_val}')
measure_nll = nn.NLLLoss()

nll = measure_nll(pred.log(), targets)
ece, mce, sce = get_calib(pred, targets)

print(f'nll {nll}, and ece {ece, mce, sce}')


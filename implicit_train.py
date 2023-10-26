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
import yaml
import numpy as np
import argparse
import pickle
import math
#from torch.cuda import amp #TODO this cannot help as SVD is not implemented with amp

# follows layout of laplace-redux https://github.com/runame/laplace-redux/blob/main/baselines/bbb/train.py

# 784*400 + 400 + 160000 + 400 + 4010 = 478,410 MLP parameters
# 478500 / 500 = 957

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'])
parser.add_argument('--model', default='MLP', choices=['MLP', 'LeNet','resnet'])
parser.add_argument('--noisy_lb', type=bool, default=False)
# parser.add_argument('--var0', type=float, default=1, help='Gaussian prior variance. If None, it will be computed to
# emulate weight decay')
parser.add_argument('--randseed', type=int, default=123)
parser.add_argument('--accurate', type=bool, default=False)
parser.add_argument('--train_samples', type=int, default=10)
parser.add_argument('--train_batch_size',type=int,default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--ind', type=bool, default=False)
parser.add_argument('--lr', type=float, default=0.7e-4)
parser.add_argument('--config',type=str,default='cifar100_gen_arch')
parser.add_argument('--use_trained',type=bool,default=False)
parser.add_argument('--train_from',type=str,default='cifar100_2123_noisy_lb_arch')
parser.add_argument('--small_noise',type=float, default=0.001)
args = parser.parse_args()

np.random.seed(args.randseed)
tr.manual_seed(args.randseed)
# tr.backends.cudnn.deterministic = True
# tr.backends.cudnn.benchmark = True

# Just symlink your dataset folder into your home directory like so
# No need to change this code---this way it's more consistent

data_path = os.path.expanduser('data/')

if tr.cuda.is_available():
    device = 'cuda:0'

if args.dataset == 'MNIST':
    train_loader, val_loader, test_loader = data_loaders.get_mnist_loaders(data_path, batch_size=args.train_batch_size,download=True)
elif args.dataset == 'CIFAR10':
    train_loader, val_loader, test_loader = data_loaders.get_cifar10_loaders(data_path,train_batch_size=args.train_batch_size,download=True)
else:
    train_loader, val_loader, test_loader = data_loaders.get_cifar100_loaders(data_path, batch_size=args.train_batch_size,
                                                                              train_batch_size=args.train_batch_size, download=True)

targets = tr.cat([y for x, y in test_loader], dim=0).numpy()
num_classes = 100 if args.dataset == 'CIFAR100' else 10
# if args.var0 is None:
#     args.var0 = 1/(5e-4*len(train_loader.dataset))

# if args.dataset == 'MNIST':
#     model = LeNetBBB(num_classes, var0=args.var0, estimator=args.estimator)
#     optmz = opt.Adam(model.parameters(), lr=1e-3, weight_decay=0)
#     arch_name = 'lenet'

dir_name = f'{"accurate" if args.accurate else "dlb"}_{args.dataset}_{args.model}'

print(f'save dir name: {dir_name} , ELBO samples: {args.train_samples}, seed: {args.randseed}')

if args.use_trained:
    load_path = f'./saved_models/{dir_name}/{args.train_from}'
# else:
#     model = WideResNetBBB(16, 4, num_classes, var0=args.var0, estimator=args.estimator)
#     optmz = opt.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0, nesterov=True)
#     arch_name = 'wrn_16-4'
#     dir_name = 'wrn_16-4_cifar10d'

# this script right now only supports MLPs for MNIST and CIFAR10.

# model specifications below:

input_shape = next(iter(train_loader))[0].shape  # FIXME test this first, not final, 785 or 784
input_shape = math.prod(input_shape[1:])

if args.dataset == 'MNIST':
    if args.model == 'MLP':
        generator = MMgenerator(input_features=[40, 40], output_features=[957, 500], n_hidden=2, hidden_units=420, activ_func=nnj.ELU)

        # FIXME parameter matching req, manual calculation is pain
        bnn_clf = BayesianMLP(input_dim=input_shape, hidden_units=[400, 400, 10], generator_network=generator,
                              likelihood='classification', device=device)

    else:
        generator = MMgenerator(input_features=[65, 65], output_features=[350, 127], n_hidden=1, hidden_units=250,
                                activ_func=nnj.ELU)
        bnn_clf = BayesianLeNet(generator)
else:
    if args.ind:
        with open('cifar_gen_arch2.yaml','r') as f:
            config = yaml.full_load(f)
        input_f, output_features, hidden_units, n_hid = config
        gen_list = 3*[CorreMMGenerator(input_features=input_f, output_features=output_features,
                                      hidden_units=hidden_units, n_hidden=n_hid)]
        gen_list = [gen.to(device) for gen in gen_list]
        bnn_clf = BayesianWideResNet(gen_list, 16, 4, 10)
        bnn_clf.to(device)
        final_out = 3*sum(math.prod(v) for v in output_features.values())
    else:
        if args.dataset == 'CIFAR10':
            #cifar_gen_architecture.yaml
            with open(f'gen_arch_config/{args.config}.yaml', 'r') as cnfg_f:
                config = yaml.full_load(cnfg_f)
            input_f, output_features, hidden_units, n_hid = config
            generator = CorreMMGenerator(input_features=input_f, output_features=output_features, hidden_units=hidden_units,
                                         n_hidden=n_hid,activ_func=nnj.ELU)
            generator.to(device)
            bnn_clf = BayesianWideResNet(generator, 16, 4, 10)
            bnn_clf.to(device)
            final_out = sum(math.prod(v) for v in output_features.values())
        else:
            with open(f'gen_arch_config/{args.config}.yaml', 'r') as cnfg_f:
                config = yaml.full_load(cnfg_f)
            input_f, output_features, hidden_units, n_hid = config
            generator = CorreMMGenerator(input_features=input_f, output_features=output_features, hidden_units=hidden_units,
                                         n_hidden=n_hid, activ_func=nnj.ELU)
            generator.to(device)
            #breakpoint()
            bnn_clf = BayesianWideResNet(generator, 28, 10, 100)
            #final_out = bnn_clf.w_dict['param_index']
            bnn_clf.to(device)
            final_out = sum(math.prod(v) for v in output_features.values())


#FIXME is there any benefit to keeping the base a multivariate normal but having a diag covariance?
# base_dist = dist.MultivariateNormal(loc=tr.zeros(bnn_clf.noise_dim, device=device),
#                                     covariance_matrix=tr.diag(tr.ones(bnn_clf.noise_dim, device=device)))

base_dist = dist.Normal(loc=tr.zeros(bnn_clf.noise_dim, device=device), scale=tr.ones(bnn_clf.noise_dim,device=device))

if args.dataset == 'MNIST':
    prior_dist = dist.Normal(loc=tr.zeros(math.prod(generator.list_layers[-1].out_features), device=device),
                             scale=tr.ones(math.prod(generator.list_layers[-1].out_features), device=device))
else:
    prior_dist = dist.Normal(loc=tr.zeros(final_out, device=device),
                             scale=tr.ones(final_out, device=device))


imp_vi_cls = ImplicitBNNs(bnn_clf, prior_dist, base_dist,args.small_noise)
if args.use_trained:
    imp_vi_cls.bnn.load_state_dict(tr.load(load_path))
if args.ind:
    optmz = opt.Adam(list(imp_vi_cls.bnn.gen_net[0].parameters())+list(imp_vi_cls.bnn.gen_net[1].parameters())+list(imp_vi_cls.bnn.gen_net[2].parameters()),lr=args.lr)
else:
    #param_list = []
    #for p in range(len(imp_vi_cls.bnn.gen_net.list_gen)):
    #    param_list.extend(list(imp_vi_cls.bnn.gen_net.list_gen[p].parameters()))
    optmz = opt.Adam(imp_vi_cls.bnn.parameters(), lr=args.lr, amsgrad=True)
    #optmz = opt.SGD(imp_vi_cls.bnn.parameters(),lr=args.lr,momentum=0.9)


# this would return generator params as they are trainable
print(f'Num. trainable/generator params for {args.model} bnn: {sum(p.numel() for p in bnn_clf.parameters() if p.requires_grad):,}')

if tr.cuda.is_available():
    imp_vi_cls.bnn.cuda()
    #print(f'model on gpu:{next(imp_vi_cls.bnn.parameters()).device}')
imp_vi_cls.bnn.train()

n_epochs = args.epochs
pbar = trange(n_epochs)
## T_max is the max iterations: n_epochs x n_batches_per_epoch
scheduler = opt.lr_scheduler.CosineAnnealingLR(optmz, T_max=n_epochs * len(train_loader),eta_min=1e-9) #eta_min=1e-8
# scheduler = opt.lr_scheduler.StepLR(optmz,step_size=500,gamma=0.5))

## For automatic-mixed-precision
# scaler = amp.GradScaler()

# Timing stuff
timing_start = tr.cuda.Event(enable_timing=True)
timing_end = tr.cuda.Event(enable_timing=True)
tr.cuda.synchronize()
timing_start.record()

#breakpoint()
for epoch in pbar:
    train_loss = 0
    num_data = len(train_loader.dataset)

    for batch_idx, (x, y) in enumerate(train_loader):
        imp_vi_cls.bnn.train()
        optmz.zero_grad()

        m = len(x)  # Batch size
        x, y = x.cuda(non_blocking=True), y.long().cuda(non_blocking=True)

        # with amp.autocast():
        # jacobian down weighing added as this is minibatched loss
        loss, ll_ls, H = imp_vi_cls.full_batch_reparam_elbo([x, y], n_samples=args.train_samples,
                                                  jacobi_down_weight=1 / m, accurate=args.accurate,
                                                            prob_down_weight=(8.5e-4/ m))
        # scaler.scale(loss).backward()
        loss.backward()
        # scaler.step(optmz)
        optmz.step()
        # scaler.update() #refer to https://pytorch.org/docs/stable/amp.html#cuda-ops-that-can-autocast-to-float16
        scheduler.step()

        train_loss = 0.9 * train_loss + 0.1 * loss.item()  # FIXME why report this?

    imp_vi_cls.bnn.eval()
    pred = imp_vi_cls.predict(test_loader, n_samples=20).cpu().numpy()
    acc_val = np.mean(np.argmax(pred, 1) == targets) * 100
    mmc_val = pred.max(-1).mean() * 100

    pbar.set_description(
        f'[ELBO: {train_loss:.3f},{ll_ls:.3f},{H:.3f}; acc: {acc_val:.1f}; mmc: {mmc_val:.1f}]'
    )

# Timing stuff
timing_end.record()
tr.cuda.synchronize()
timing = timing_start.elapsed_time(timing_end) / 1000
np.save(f'result_log/timings_train/imp_vi-{args.accurate}-{args.model}_{args.dataset.lower()}_{args.randseed}_{"noisy_lb" if args.noisy_lb else "_"}n', timing)

path = f'./saved_models/{dir_name}'

if not os.path.exists(path):
    os.makedirs(path)

if not args.use_trained:
    save_name = f'{path}/{args.dataset.lower()}_{args.randseed}_{"noisy_lb" if args.noisy_lb else "_"}_{args.config.split("_")[-1]}'
else:
    save_name = f'{path}/{args.dataset.lower()}_{args.randseed}_{"noisy_lb" if args.noisy_lb else "_"}_{args.config.split("_")[-1]}_retrained'

dump_dict = dict()
dump_dict['model_weights'] = imp_vi_cls.bnn.state_dict()
dump_dict['small_noise'] = imp_vi_cls.small_noise
tr.save(dump_dict, save_name)
del dump_dict #removing from memory as this could be heavy
#imp_vi_cls.bnn.cpu()
## Try loading and testing
load_dict = tr.load(save_name)
imp_vi_cls.bnn.load_state_dict(load_dict['model_weights'])
imp_vi_cls.bnn.eval()

## In-distribution
py_in = imp_vi_cls.predict(test_loader, n_samples=40).cpu().numpy()
acc_in = np.mean(np.argmax(py_in, 1) == targets) * 100
print(f'Accuracy: {acc_in:.1f}')

from math import fabs
import torch as tr
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from stochman import nnj
import torch.nn as nn
from torch import Tensor
from typing import Union
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.distributions as dist
from typing import Union
import pickle


# TODO return underlying function from the toy data generators

def decay_sin(start, num_data_points, end, sin_mul_factor=0.1, decay='inv', heteroskedasticity=False, astensor=True):
    #     end = start + num_data_points * sample_resl
    x = np.linspace(start, end, num_data_points, dtype=np.float32)
    x_test = np.linspace(start + 0.02, end - 1.0, int(num_data_points * 0.25)).astype(np.float32)
    cns = 30.
    sbt = 2 * x.mean()
    if decay == "inv":
        if heteroskedasticity:
            y = (cns / np.abs(x - sbt)) * np.sin(x * sin_mul_factor) + (1 / x) * np.random.randn(x.size)
            y = y.astype(np.float32, copy=False)
            y_test = (cns / np.abs(x_test - sbt)) * np.sin(x_test * sin_mul_factor)
            y_test = y_test.astype(np.float32, copy=False)
        else:
            y = (cns / np.abs(x - sbt)) * np.sin(x * sin_mul_factor) + 0.01 * np.random.randn(x.size)
            y = y.astype(np.float32, copy=False)
            y_test = (cns / np.abs(x_test - sbt)) * np.sin(x_test * sin_mul_factor)
            y_test = y_test.astype(np.float32, copy=False)
    elif decay == "sqrt inv":
        if heteroskedasticity:
            y = (cns / (x ** 0.5)) * np.sin(x * sin_mul_factor) + (1 / (x ** 0.5)) * np.random.randn(x.size)
            y = y.astype(np.float32, copy=False)
            y_test = (cns / (x_test ** 0.5)) * np.sin(x_test * sin_mul_factor)
            y_test = y_test.astype(np.float32, copy=False)
        else:
            y = (cns / (x ** 0.5)) * np.sin(x * sin_mul_factor) + 0.01 * np.random.randn(x.size)
            y = y.astype(np.float32, copy=False)
            y_test = (cns / (x_test ** 0.5)) * np.sin(x_test * sin_mul_factor)
            y_test = y_test.astype(np.float32, copy=False)
    else:
        raise NotImplementedError
    if astensor:
        y = tr.from_numpy(y)
        x = tr.from_numpy(x)
        y_test = tr.from_numpy(y_test)
        x_test = tr.from_numpy(x_test)

    return y[:, np.newaxis], x[:, np.newaxis], y_test[:, np.newaxis], x_test[:, np.newaxis]


def sum_sinusoid(start: float, num_data_points: int, end: float, eps=1e-6, astensor: bool = True,
                 func_return: Union[bool, Tensor, np.ndarray] = False):

    if isinstance(func_return, Tensor) or isinstance(func_return, np.ndarray):
        y = func_return + 0.8 * np.sin(2 * np.pi * (func_return + eps)) + 0.8 * np.sin(
            4 * np.pi * (func_return + eps)) + eps
        return y
    elif func_return:
        x = np.linspace(start=start, stop=end, num=num_data_points, dtype=np.float32)
        x_test = np.linspace(start=start, stop=end, num=int(num_data_points * 0.2)).astype(
            np.float32)  # 20 percent of the train data
        y = x + 0.8 * np.sin(2 * np.pi * (x + eps)) + 0.8 * np.sin(4 * np.pi * (x + eps)) + eps
        return y

    else:
        x = np.linspace(start=start, stop=end, num=num_data_points, dtype=np.float32)
        x_test = np.linspace(start=start, stop=end, num=int(num_data_points * 0.2)).astype(
            np.float32)  # 20 percent of the train data
        y = x + 0.8 * np.sin(2 * np.pi * (x + eps)) + 0.8 * np.sin(4 * np.pi * (x + eps)) + eps + 0.3 * np.random.randn(
            *x.shape)
        y = y.astype(np.float32, copy=False)
        y_test = x_test + 0.8 * np.sin(2 * np.pi * (x_test + eps)) + 0.8 * np.sin(4 * np.pi * (x_test + eps)) + eps
        y_test = y_test.astype(np.float32, copy=False)

        if astensor:
            y = tr.from_numpy(y)
            x = tr.from_numpy(x)
            y_test = tr.from_numpy(y_test)
            x_test = tr.from_numpy(x_test)
        return y[:, np.newaxis], x[:, np.newaxis], y_test[:, np.newaxis], x_test[:, np.newaxis]


def plot_data(y, x):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel("regression inputs")
    ax.set_ylabel("outputs with gaussian response noise")
    ax.set_title("Regression Data")
    plt.show()


class Atan(nnj.AbstractActivationJacobian, nn.Tanh):
    def forward(self, x: Tensor) -> Tensor:
        val = tr.atan(x)
        return val  # CHECK do they return vals in stochman? compatibility with torch?

    def _jacobian(self, x: Tensor, val: Tensor) -> Tensor:
        jac = 1.0 / (1.0 + x ** 2)
        return jac


def softclip(tensor, min):
    """Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials"""
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


@tr.no_grad()
def avbpredict(bnn, dataloader, n_samples=100):
    py = []
    # sample once for all predictions
    device = 'cuda:0' if tr.cuda.is_available() else 'cpu'


    for x, y in dataloader:
        base_sample, params, weights_to_layers = bnn.sample_params(batch_size=n_samples, device=device) #resampling the parameters for every minibatch
        #sampling repeatedly for each minibatch should reduce conf

        x = x.cuda() if tr.cuda.is_available() else x

        _py = 0
        for s in range(n_samples):
            weights_to_layers(params[s, :])
            f_s = bnn(x)  # The second return is KL
            # if self.bnn.named_children()
            _py += tr.softmax(f_s, 1)
        _py /= n_samples

        py.append(_py)

    return tr.cat(py, dim=0)


def mfvipredict(bnn, test_loader, n_samples=100):
    device = 'cuda:0' if tr.cuda.is_available() else 'cpu'
    num_samples = n_samples
    with tr.no_grad():
        pred_probs_mc = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred_probs_mc_batch = []
            for mc_run in range(num_samples):
                bnn.eval()
                output = bnn.forward(data)
                #get probabilities from log-prob
                pred_probs = tr.softmax(output,-1)
                pred_probs_mc_batch.append(pred_probs.cpu().unsqueeze(0))
            
            pred_probs_mc.append(tr.mean(tr.cat(pred_probs_mc_batch,dim=0),dim=0))

        target_labels = target.cpu().data.numpy()
        pred_mean = tr.cat(pred_probs_mc)
        Y_pred = np.argmax(pred_mean, axis=-1)
        return pred_mean, Y_pred

    
def mnfpredict(mnf_net, test_loader, n_samples=100):
    device = 'cuda:0' if tr.cuda.is_available() else 'cpu'
    num_samples = n_samples
    with tr.no_grad():
        pred_probs_mc = []
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred_probs_mc_batch = []
            for mc_run in range(num_samples):
                mnf_net.eval()
                output = mnf_net(data)
                #MNFLeNet has log softmax in the last layer 
                pred_probs = output.exp().detach()
                pred_probs_mc_batch.append(pred_probs.cpu().unsqueeze(0))
            
            pred_probs_mc.append(tr.mean(tr.cat(pred_probs_mc_batch,dim=0),dim=0))

        target_labels = target.cpu().data.numpy()
        pred_mean = tr.cat(pred_probs_mc)
        Y_pred = np.argmax(pred_mean, axis=-1)
        return pred_mean, Y_pred

    

data_std_hyparam = 0.1



def plotting_code(y: Tensor, x: Tensor, y_test: Tensor, x_test: Tensor, imp_bnn,  gaussian_noise_ll: Tensor, save: bool=False,
                   device='cpu', plotting_samples: int=20,standardised:bool=False, data_max:float=0.5):
    # putting both network in eval
    imp_bnn.bnn.eval()
    imp_bnn.bnn.gen_net.eval()
    #plot the regression function or decision boundary outside of the training data range.
    plot_x = tr.linspace(x.min()-0.65,x.max()+0.65,x.shape[0] + x_test.shape[0] + 40)[:,None].to(device)

    def mean_predictive(n_samples=10):
        with tr.no_grad():
            base_samp, params, w_to_l = imp_bnn.bnn.sample_params(batch_size=n_samples, device=device)
            noisy_params = params + imp_bnn.small_noise * tr.randn_like(params)
            preds = []
            for ns in range(n_samples):
                #insert params into layers
            #             if params[ns, :].max() > 1.0:
            #                 w_to_l(params[ns, :] / params[ns, :].max())  # set params inside
            #             else:
                w_to_l(noisy_params[ns, :])
                test_out = imp_bnn.bnn(plot_x)
                preds.append(test_out.detach())
            mean_pred = tr.stack(preds).mean(axis=0)
            mixed_noise = tr.stack(preds) + gaussian_noise_ll*tr.randn_like(tr.stack(preds))
            conf_intrvl = np.percentile(tr.stack(preds).cpu().numpy(), [5.0, 95.0], axis=0)
            mixed_noise_intrvl = np.percentile(mixed_noise.cpu().numpy(), [5.0,95.0], axis=0)
            return mean_pred, conf_intrvl, mixed_noise_intrvl
    mean_pred, conf_intrvl, mixed_noise_intrvl = mean_predictive(n_samples=plotting_samples)
    fig, ax = plt.subplots(1,2,figsize=(18,7))
    fontdict = {'fontsize':24}
    true_func = sum_sinusoid(x.min(),len(x),x.max(),func_return=plot_x.cpu())
    if standardised:
        plot_x_std = plot_x/data_max
    ax[0].fill_between(plot_x.squeeze().cpu().numpy(), conf_intrvl[0, :][..., 0], conf_intrvl[1, :][..., 0],
                    color='lightblue', label='model noise',alpha=0.9)
    ax[0].fill_between(plot_x.squeeze().cpu().numpy(), mean_pred.squeeze().cpu().numpy()-2*gaussian_noise_ll.cpu().numpy(),mean_pred.squeeze().cpu().numpy()+2*gaussian_noise_ll.cpu().numpy(),color='lightpink', label='data noise',alpha=0.45)
    ax[0].scatter(x.cpu().numpy(), y.cpu().numpy(),marker="+", color='k', label='training data')
    ax[0].scatter(x_test.squeeze().cpu().numpy(), y_test.cpu().squeeze().numpy(), label='test data')
    ax[0].plot(plot_x.squeeze().cpu().numpy(), mean_pred.squeeze().cpu().numpy(), 'r', label='mean predictive')
    if standardised:
        ax[0].plot(plot_x_std.squeeze().cpu().numpy(), true_func,"k",marker="_", label="Ground truth")
    else:
        ax[0].plot(plot_x.squeeze().cpu().numpy(), true_func,"k",marker="_",label="Ground truth")
    #y_limit = ax[0].get_ylim()
    y_limit=(-8.0,16)
    ax[0].set_ylim(y_limit)
    ax[0].legend()
    ax[0].set_title('ImpVI Data and model noise',fontdict=fontdict)
    ###
    ax[1].fill_between(plot_x.squeeze().cpu().numpy(), mixed_noise_intrvl[0, :][..., 0], mixed_noise_intrvl[1, :][..., 0],
                    color='lightblue', label='mixed noise interval')
    ax[1].scatter(x.cpu().numpy(), y.cpu().numpy(),marker="+", color='k', label='training data')
    ax[1].scatter(x_test.squeeze().cpu().numpy(), y_test.cpu().squeeze().numpy(), label='test data')
    ax[1].plot(plot_x.squeeze().cpu().numpy(), mean_pred.squeeze().cpu().numpy(), 'r', label='mean predictive')
    if standardised:
        ax[1].plot(plot_x_std.squeeze().cpu().numpy(), true_func,"k", label="Ground truth")
    else:
        ax[1].plot(plot_x.squeeze().cpu().numpy(), true_func,"k", label="Ground truth")
    ax[1].set_title('ImpVI Mixed uncertainty',fontdict=fontdict)
    ax[1].set_ylim(y_limit)
    ax[1].legend()
    ####
#     avg_pred, conf_interval = logger_dict2['hmc_bnn']
#     plot_x = logger_dict2['plot_x']
#     ax[2].plot(plot_x.cpu().numpy(), avg_pred.cpu().numpy(),'r',l|abel='mean predictive')
#     ax[2].fill_between(plot_x.squeeze().cpu().numpy(), conf_interval[0, :][..., 0], conf_interval[1, :][..., 0],
#                 color='lightblue', label='conf interval')
#     ax[2].scatter(x_test.cpu().numpy(), y_test.cpu().numpy(), marker="x",label='test points')
#     ax[2].scatter(train_x.cpu().numpy(), train_y.cpu().numpy(), marker="+", color='k',label='train points')
#     ax[2].set_ylim(y_limit)
#     ax[2].set_title('HMC sampled posterior',fontdict=fontdict)
#     ax[2].set_title('HMC')
    plt.show()
    if save:
        fig.savefig("toy_plt.pdf",format='pdf')
    return mean_pred, conf_intrvl, mixed_noise_intrvl #added for logging

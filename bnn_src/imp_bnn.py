import torch as tr
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from bnn_src.models import SimpleMLP, MMgenerator
import matplotlib.pyplot as plt
import torch.distributions as dist
from bnn_src.models import BayesianMLP
#from torchviz import make_dot
import warnings
import math
from typing import Union


def create_generator_nn(base_samp_dim: int, output_param_dim, hidden_units, activ_func=nn.ReLU):
    assert hidden_units[-1] == output_param_dim, 'The generator architecture ' \
                                                 'specifications are incorrect'
    # assert activ_func.__bases__ == "nn.modules.module.Module"       doesn't work
    if hidden_units[-1] == 1:
        print('Are you doing regression?')
    bnn_gen = SimpleMLP(input_dim=base_samp_dim, hidden_units=hidden_units, activ_func=activ_func, jac_req=True)
    return bnn_gen


class ImplicitBNNs:
    def __init__(self, bnn: BayesianMLP, prior_dist: Union[dist.MultivariateNormal, SimpleMLP, dist.Normal, MMgenerator],
                 base_dist: Union[dist.MultivariateNormal, dist.Normal], small_noise: int=0.001):
        self.bnn = bnn
        self.prior_dist = prior_dist
        self.base_dist = base_dist
        self.likelihood = bnn.likelihood
        self.small_noise = small_noise

    def full_batch_reparam_elbo(self, pred_data: Union[list, tuple], likelihood_loss = tr.nn.CrossEntropyLoss, n_samples: int = 10,
                                jacobi_down_weight: float = 0.1, accurate: bool = True, prob_down_weight: float = 1e-2):
        assert n_samples > 0, 'number of samples cannot be negative integer'
        device = 'cuda:0' if tr.cuda.is_available() else 'cpu' # if hasattr(self.bnn, 'device') else 'cpu'
        elbo = []
        data_len = pred_data[0].shape[0]
        assert self.likelihood == 'classification', 'function only tested for classification'
        loss_fn = likelihood_loss() #default reduction in the loss is 'mean'
        # TODO insert an appropriate device check
        #     if device != 'cpu':
        #         assert next(bnn.parameters()).device == device, "device inconsistency detected in elbo loss"
        # base_sample.requires_grad_() #FIXME IS THIS NEEDED HERE?
        self.bnn.train()
        base_s, params, weights_to_layers = self.bnn.sample_params(batch_size=n_samples, device=device)
        if isinstance(base_s, list):
            base_sample = tr.cat(base_s, dim=1)
        else:
            base_sample = base_s
        params = params + self.small_noise * tr.randn_like(params)
        # try a heuristic down weighing of the weights initially to avoid the high likelihood loss/high gradient issue

        for ns in range(n_samples):  # mean across noise samples.
            # if tr.abs(params[ns, :].detach()).max() > 3.0:
            #     weights_to_layers(params[ns, :] / tr.abs(params[ns, :].detach()).max()) #set params inside
            # else:
            weights_to_layers(params[ns, :])  # do i need downscaling when the gradients are dying?
            # likelihood loss function is over the whole batch, no normalization w.r.t data points needed
            likelihood_ls = -loss_fn(self.bnn(pred_data[0]), pred_data[1])
            
            single_samp_lb = likelihood_ls - prob_down_weight * tr.sum(self.base_dist.log_prob(base_sample[ns, :])) + \
                             prob_down_weight * tr.sum(self.prior_dist.log_prob(params[ns, :]))
            # TODO downweigh the entropy of single samp elbo??
            elbo.append(single_samp_lb)

        if isinstance(self.bnn.gen_net, list):
        # log(q) calculation with independent weights assumption
            jacobi_list = []
            for bs, gen in zip(base_s, self.bnn.gen_net):
                jacobi_list.append(self.entropy_lb(gen, base_sample=bs, device=device,
                                          singular_value_iter=5, accurate=accurate))
            accurate_jacobi = tr.sum(tr.as_tensor(jacobi_list))
        else:
            accurate_jacobi = self.entropy_lb(self.bnn.gen_net, base_sample=base_sample, device=device,
                                          singular_value_iter=3, accurate=accurate)
        # print('value of J^t.J: ',accurate_jacobi.item())
        # down weight the accurate jacobi term. accurate jacobi increases with iterations why? Do we need more
        # regularisation? ADDED downweigh by data cardinality
        mc_estimate_elbo = tr.stack(elbo).mean() + jacobi_down_weight * (accurate_jacobi / data_len)

        #   E[log(p(D|G(z))) + log(p(G(z))) - log(p_0(G(z)))] + E[lower_bound(log(q(z)))]
        # elbo is lower bound to ML and needs to increase so the loss reported is the negative of the elbo
        return -mc_estimate_elbo, likelihood_ls.detach(), accurate_jacobi.detach()

    ##training with NLL, can't I just use torch NLL, TODO

    def full_batch_reparam_elbo_nll(self, pred_data: Union[list, tuple], device='cuda:0', n_samples: int = 10,
                                    jacobi_down_weight: float = 0.1, accurate: bool = True, prob_down_weight: float = 1e-2):
        
        assert self.bnn.likelihood == 'regression', 'This loss function only works for regression'
        assert n_samples > 0, 'number of samples can be positive non-zero integer'
        device = device if tr.cuda.is_available() else "cpu"
        elbo = []
        data_len = pred_data[0].shape[0]
        # TODO insert an appropriate device check
        #     if device != 'cpu':
        #         assert next(bnn.parameters()).device == device, "device inconsistency detected in elbo loss"
        # base_sample.requires_grad_() #FIXME IS THIS NEEDED HERE?
        self.bnn.train()
        base_sample, params, weights_to_layers = self.bnn.sample_params(batch_size=n_samples, device=device)
        params = params + self.small_noise * tr.randn_like(params)
        gaussian_noise_ll = tr.exp(0.5 * self.bnn.ll_log_var) + 1e-8
        # gaussian_noise_ll = (self.bnn.gen_net.ll_var)**0.5 + 1e-7
        for ns in range(n_samples):  # mean across noise samples.
            weights_to_layers(params[ns, :])  # do i need downscaling when the gradients are dying?
            # likelihood loss function is over the whole batch, no normalization w.r.t data points needed
            data_ll = dist.Normal(self.bnn(pred_data[0]), gaussian_noise_ll).log_prob(pred_data[1]).sum()
            single_samp_lb = data_ll - \
                             prob_down_weight * self.base_dist.log_prob(base_sample[ns, :]) + \
                             prob_down_weight * self.prior_dist.log_prob(params[ns, :])
            # TODO downweigh the entropy of single samp elbo??
            elbo.append(single_samp_lb)

        accurate_jacobi = self.entropy_lb(self.bnn.gen_net, base_sample=base_sample, device=device,
                                          singular_value_iter=5, accurate=accurate)
        # print('value of J^t.J: ',accurate_jacobi.item())

        mc_estimate_elbo = tr.stack(elbo).mean() + jacobi_down_weight * (accurate_jacobi)

        #   E[log(p(D|G(z))) + log(p(G(z))) - log(p_0(G(z)))] + E[lower_bound(log(q(z)))]
        # elbo is lower bound to ML and needs to increase so the loss reported is the negative of the elbo
        return -mc_estimate_elbo, data_ll.detach(), accurate_jacobi.detach()

    def full_batch_relbo_cl(self, pred_data: Union[list, tuple], device='cuda:0', n_samples: int = 10,
                            jacobi_down_weight: float = 0.1,
                            accurate: bool = True,
                            prob_down_weight: float = 1e-2):

        assert isinstance(self.prior_dist, SimpleMLP), 'Continual learning only supported for implicit posteriors ' \
                                                       'using generators'
        assert self.bnn.likelihood == 'regression', 'This loss function only works for regression'
        assert n_samples > 0, 'number of samples can be positive non-zero integer'
        device = device if tr.cuda.is_available() else "cpu"
        assert accurate, 'right now this only works for accurate jac compuatation'
        elbo = []
        data_len = pred_data[0].shape[0]

        # TODO insert an appropriate device check
        #     if device != 'cpu':
        #         assert next(bnn.parameters()).device == device, "device inconsistency detected in elbo loss"
        # base_sample.requires_grad_() #FIXME IS THIS NEEDED HERE?
        # only forward pass jacobian evals using stochman nnj
        for p in self.prior_dist.parameters():
            p.requires_grad = False
        self.bnn.train()
        base_sample, params, weights_to_layers = self.bnn.sample_params(batch_size=n_samples, device=device)

        prob_down_weight = prob_down_weight
        # gaussian_noise_ll = tr.exp(0.5 * self.bnn.gen_net.ll_log_var) + 1e-8
        gaussian_noise_ll = (self.bnn.gen_net.ll_var) ** 0.5 + 1e-7
        for ns in range(n_samples):  # mean across noise samples.
            weights_to_layers(params[ns, :])  # do i need downscaling when the gradients are dying?
            # likelihood loss function is over the whole batch, no normalization w.r.t data points needed
            data_ll = dist.Normal(self.bnn(pred_data[0]), gaussian_noise_ll).log_prob(pred_data[1]).sum()
            single_samp_lb = data_ll - prob_down_weight * self.entropy_lb(self.prior_dist, base_sample, device=device,
                                                                          singular_value_iter=5, accurate=accurate)
            # TODO downweigh the entropy of single samp elbo??
            elbo.append(single_samp_lb)

        accurate_jacobi = self.entropy_lb(self.bnn.gen_net, base_sample=base_sample, device=device,
                                          singular_value_iter=5, accurate=accurate)
        # print('value of J^t.J: ',accurate_jacobi.item())

        mc_estimate_elbo = tr.stack(elbo).mean() + jacobi_down_weight * (accurate_jacobi)

        #   E[log(p(D|G(z))) + log(p(G(z))) - log(p_0(G(z)))] + E[lower_bound(log(q(z)))]
        # elbo is lower bound to ML and needs to increase so the loss reported is the negative of the elbo
        return -mc_estimate_elbo, data_ll.detach(), accurate_jacobi.detach()

    def entropy_lb(self, generator, base_sample, device='cuda:0', singular_value_iter=12, accurate: bool = False):
        # smallest singluar value approximation
        def RaRitz(v, r, p, intermediate, projection, device):
            JV = []
            r = r / (r.mean(list(range(1,r.ndim)), True))
            if p.norm(2, list(range(1,p.ndim))).min() == 0:
                p = tr.randn_like(p)
            #print('shape print in raritz: ',r.shape, p.shape,v.shape)
            V = tr.stack((v, r, p), -1)
            try:
                V = tr.svd(V).U
            except Exception as e:
                # print(e)
                V = tr.randn(V.shape).to(device)
                V = tr.svd(V).U

            for i in range(V.shape[-1]):
                Jv = tr.autograd.grad(intermediate, projection, V[..., i], retain_graph=True)[0]
                JV.append(Jv)

            if len(JV[0].shape) == 4:
                JV = tr.stack(JV, -1).flatten(1, 3)
            else:
                JV = tr.stack(JV, -1)
            V = V.reshape(V.shape[0],-1,V.shape[-1])
            v_min = tr.svd(JV).V[..., -1:].cuda() if tr.cuda.is_available() else tr.svd(JV).V[...,-1:]
            p_op = V[...,-2:].bmm(v_min[:, -2:]).squeeze(-1)
            p_op = p_op.reshape(p.shape)
            p_op_norm = p_op.norm(2, dim=list(range(1,p.ndim)))
            p.data.index_copy_(0, tr.where(~p_op_norm.isnan())[0], p_op[~p_op_norm.isnan()].detach())
            v_op = V.bmm(v_min).squeeze(-1)
            v_op = v_op.reshape(v.shape)
            v_op_norm = v_op.norm(2, dim=list(range(1,p.ndim)))
            v.data.index_copy_(0, tr.where(~v_op_norm.isnan())[0], v_op[~v_op_norm.isnan()].detach())
            return v, p

        def compute_entropy_s(gen, z, device=device):
            z.requires_grad_()
            gen.eval()
            fake_eval = gen(z)
            projection = tr.ones_like(fake_eval, requires_grad=True).to(device)
            intermediate = tr.autograd.grad(fake_eval, z, projection, create_graph=True)
            v = tr.randn(z.shape).to(device)
            p = tr.randn(z.shape).to(device)
            Jv = tr.autograd.grad(intermediate[0], projection, v, retain_graph=True)[0]
            size = len(Jv.shape)
            # Jv=J.bmm(self.v.unsqueeze(-1)).squeeze()
            mu = Jv.norm(2, dim=list(range(1, size))) / v.norm(2, dim=list(range(1,v.ndim)))
            for i in range(singular_value_iter):
                JTJv = (Jv.detach() * fake_eval).sum(list(range(1, size)), True)
                for _ in range(v.ndim-1):
                    mu.unsqueeze_(-1)
                r = tr.autograd.grad(JTJv, z, tr.ones_like(JTJv, requires_grad=True), retain_graph=True)[0] \
                    - (mu ** 2) * v
                v, p = RaRitz(v, r, p, intermediate[0], projection, device)
                Jv = tr.autograd.grad(intermediate[0], projection, v.detach(), create_graph=True)[0]
                mu = Jv.norm(2, dim=list(range(1, size)))
            est = math.prod(z.shape[1:]) * tr.log(mu + self.small_noise**2)
            H = est.mean()
            gen.train()
            z.requires_grad = False  # turn off gradients for the base sample.
            return H

        # accurate jacobian calculation
        def compute_entropy_acc(gen, z, eps=1e-5):
            # batchnorm here requires .eval()
            # no requirement of requires_grad = True
            gen.eval()
            _, J = gen(z, True)
            J = J.reshape(z.shape[0], -1, z.shape[-1])
            #jtj = tr.bmm(tr.transpose(J, -2, -1), J)
            jjt = tr.bmm(J, tr.transpose(J,-2,-1))
            H = 0.5 * tr.slogdet(jjt + eps*tr.eye(jjt.shape[-1], device=jjt.device))[1]
            #H = 0.5 * tr.slogdet(jtj + eps)[1]
            # H[tr.isneginf(H)] = -100.  #FIXME Numerical instability
            gen.train()
            return H.mean()

        if accurate:
            return compute_entropy_acc(generator, base_sample)
        else:
            return compute_entropy_s(generator, base_sample, device=device)

    @tr.no_grad()
    def predict(self, dataloader, n_samples=1, device='cuda:0'):
        py = []
        # sample once for all predictions
        device = device
        

        for x, y in dataloader:
            base_sample, params, weights_to_layers = self.bnn.sample_params(batch_size=n_samples, device=device) #resampling the parameters for every minibatch
            noisy_params = params + self.small_noise*tr.randn_like(params)
            #sampling repeatedly for each minibatch should reduce conf
            x = x.to(device)
            #x = x.cuda() if tr.cuda.is_available() else x

            _py = 0
            for s in range(n_samples):
                weights_to_layers(noisy_params[s, :])
                f_s = self.bnn(x)  # The second return is KL
                # if self.bnn.named_children()
                _py += tr.softmax(f_s.detach(), 1)
            _py /= n_samples
            py.append(_py)

        return tr.cat(py, dim=0)
    
    @property
    def small_noise(self):
        return self._small_noise
    
    @small_noise.setter
    def small_noise(self, noise_val:int):
        assert noise_val < 0.5, 'high output noise values not currently supported'
        if noise_val > 0.1:
            warnings.warn("This is very high noise value generally not recommended")
            self._small_noise = noise_val
        else:
            self._small_noise = noise_val
        print(f'sigma for dlvm set to {noise_val}')


import torch as tr
import torch.nn as nn
import numpy as np
import sys
from typing import Union, Tuple, List, Optional
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.distributions as tr_dist
import math
import torch.nn.functional as F
import pyro
from pyro.nn.module import to_pyro_module_
from stochman import nnj
from bnn_src.layers import *
from bnn_src.utils import softclip


# The simplest neural network module there could be.
class Net(nn.Module):
    def __init__(self, input_dim=1, activ_func=nn.LeakyReLU):
        super(Net, self).__init__()
        self.input_layer = nn.Linear(input_dim, 17)
        self.fc1 = nn.Linear(17, 7)
        self.fc2 = nn.Linear(7, 1)
        # self.fc3 = nn.Linear(5,1)
        self.non_lin = activ_func()

    def forward(self, inp):
        assert inp.shape[-1] == 1, 'only for stupid regression testing'
        x = self.input_layer(inp)
        x = self.non_lin(x)
        x = self.fc1(x)
        x = self.non_lin(x)
        out = self.fc2(x)
        # out = self.non_lin(x)
        # x = self.fc3(x)
        # out = self.non_lin(x)
        return out


# The simplest generator module there could be:

class NewGen(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(NewGen, self).__init__()
        self.main = nnj.Sequential(
            nnj.Linear(input_dim, 40),
            nnj.ELU(),
            nnj.Linear(40, 40),
            nnj.ELU(),
            #           nnj.Linear(100, 100),
            #           nnj.ELU(),
            #           nnj.Linear(100, 100),
            #           nnj.ELU(),
            #           nnj.Linear(70,70),
            #           nnj.ELU(),
            nnj.Linear(40, output_dim),
            # nnj.ELU()
            # nnj.Tanh()
        )

    def forward(self, x, jac=False):
        return self.main(x, jac)


class SimpleMLP(nn.Module):

    def __init__(self, input_dim, hidden_units, activ_func=nn.ReLU, likelihood='regression',
                 jac_req=False):
        super(SimpleMLP, self).__init__()
        assert isinstance(hidden_units, list), 'hidden units improperly specified'
        hidden_layers = len(hidden_units)
        self.non_lin = activ_func
        if not jac_req:
            # if isinstance(hidden_units, list) else 0
            self.input_dim = input_dim
            first_layer = nn.Linear(self.input_dim, hidden_units[0])
            list_affine = [[self.non_lin(), nn.Linear(hidden_units[i], hidden_units[i + 1])] for i in range(
                hidden_layers - 1)]
            list_affine = [layer for lyr_active in list_affine for layer in lyr_active]
            list_affine.insert(0, first_layer)
            self.list_layers = nn.Sequential(*list_affine)
            self.likelihood = likelihood
            last_layer = nn.Linear(hidden_units[-1], 1) if self.likelihood == 'regression' and hidden_units[
                -1] != 1 else None
            if last_layer is not None:
                self.list_layers.add_module(name='non_lin', module=self.non_lin())
                self.list_layers.add_module(name='last_layer', module=last_layer)
        else:
            self.input_dim = input_dim
            self.non_lin = nnj.__dict__[str(activ_func()).split(sep='(')[0]]  # nnj method
            first_layer = nnj.Linear(self.input_dim, hidden_units[0])
            list_affine = [[self.non_lin(), nnj.Linear(hidden_units[i], hidden_units[i + 1])] for i in range(
                hidden_layers - 1)]
            list_affine = [layer for lyr_active in list_affine for layer in lyr_active]
            list_affine.insert(0, first_layer)
            self.list_layers = nnj.Sequential(*list_affine)
            self.likelihood = likelihood
            # last_layer = nn.Linear(hidden_units[-1], 1) if self.likelihood == 'regression' and hidden_units[
            #     -1] != 1 else None
            # if last_layer is not None:
            #     self.list_layers.add_module(name='non_lin', module=self.non_lin())
            #     self.list_layers.add_module(name='last_layer', module=last_layer)

    def forward(self, input: Tensor, jacobian: Union[Tensor, bool] = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        assert input.shape[-1] == self.input_dim, "input dimensions not compatible with the network"
        assert isinstance(jacobian, Tensor) or isinstance(jacobian, bool), 'jac arg is incompatible'

        if isinstance(jacobian, bool) and not jacobian:
            func_out = self.list_layers(input)
            output = nn.Softmax(func_out) if self.likelihood == 'classification' else func_out
            ## use nn.NLLloss with softmax not the nn.Crossentropyloss
            return output
        else:
            assert isinstance(self.list_layers, nnj.Sequential), 'jacobians only available for nnj'
            func_out, jac = self.list_layers(input, jacobian)
            return func_out, jac


class MMgenerator(nn.Module):

    def __init__(self, input_features: List[int], output_features: List[int], hidden_units:
    Union[int, List[int]], n_hidden: int, activ_func=nnj.ReLU):

        super(MMgenerator, self).__init__()
        if isinstance(hidden_units, int):
            hidden_units = 2 * [hidden_units]
        self.input_dim = input_features
        self.output_dim = output_features
        first_layer = MMLayer(input_features, hidden_units)
        list_hidden = [[activ_func(), MMLayer(hidden_units, hidden_units)] for _ in range(n_hidden)]
        last_layer = MMLayer(hidden_units, output_features)
        list_hidden = [layer for lyr_active in list_hidden for layer in lyr_active]
        list_hidden.insert(0, first_layer)
        list_hidden.insert(len(list_hidden), activ_func())
        list_hidden.insert(len(list_hidden), last_layer)
        self.list_layers = nnj.Sequential(*list_hidden)

    def forward(self, input, jacobian=False):
        out = self.list_layers(input, jacobian)
        if isinstance(out, tuple):
            out_f = out[0].reshape(input.shape[0], -1)
            return out_f, out[1]
        else:
            return out.reshape(input.shape[0], -1)


class CorreMLPGenerator(nn.Module):

    def __init__(self, input_features: dict, output_features: dict, hidden_units: dict, n_hidden: dict,
                 activ_func=nnj.ReLU, gen_net='MM'):
        super(CorreMLPGenerator, self).__init__()
        try:
            self.noise_input = input_features['noise_input']
        except KeyError as e:
            raise KeyError('input_features dict does not contain noise_input key')
        self.out_dim = output_features
        n_gen = gen_input_layer = 0
        self.list_gen = []
        for _ in output_features:
            n_gen += 1
        for k, v in input_features.items():
            if k != 'noise_input':
                if isinstance(v, list) or isinstance(v, tuple):
                    gen_input_layer += math.prod(v)
                else:
                    gen_input_layer += v
                hu = n_hidden[k] * [hidden_units[k]]
                hu.append(output_features[k])
                self.list_gen.append(SimpleMLP(input_dim=v, hidden_units=hu,
                                               activ_func=activ_func, jac_req=True))
        self.first_layer = nnj.Linear(input_features['noise_input'], gen_input_layer)
        self.second_layer = nnj.Linear(gen_input_layer, gen_input_layer)
        self.non_lin = activ_func()

    def forward(self, input, jacobian=False):
        if not jacobian:
            y = self.non_lin(self.first_layer(input))
            y = self.non_lin(self.second_layer(y))
            output = []
            for gn in range(len(self.list_gen)):
                index = 0
                out = self.list_gen[gn](y[:, index:index + self.list_gen[gn].input_dim])
                index += self.list_gen[gn].input_dim
                output.append(out)
            return tr.cat(output, dim=1)
        else:
            y, jac = self.first_layer(input, jacobian=jacobian)
            y_1 = self.non_lin(y)
            jac = self.non_lin._jacobian_mult(x=y, val=y_1, jac_in=jac)
            y_2 = self.second_layer(y_1)
            jac = self.second_layer._jacobian_mult(x=y_1, val=y_2, jac_in=jac)
            y_3 = self.non_lin(y_2)
            jac = self.non_lin._jacobian_mult(x=y_2, val=y_3, jac_in=jac)
            output = []
            jac_list = []
            for gn in range(len(self.list_gen)):
                index = 0
                out, jac = self.list_gen[gn](y_3[:, index: index + self.list_gen[gn].input_dim], jacobian=jac[:,
                                                                                                          index: index +
                                                                                                                 self.list_gen[
                                                                                                                     gn].input_dim,
                                                                                                          :])
                index += self.list_gen[gn].input_dim
                output.append(out)
                jac_list.append(jac)
            return tr.cat(output, dim=1), tr.cat(jac_list, dim=1)


class CorreMMGenerator(nn.Module):

    def __init__(self, input_features: dict, output_features: dict, hidden_units: dict, n_hidden: dict,
                 activ_func=nnj.ReLU):
        super(CorreMMGenerator, self).__init__()
        try:
            self.noise_input = input_features['noise_input']
        except KeyError as e:
            raise KeyError('input_features dict does not contain noise_input key')
        self.out_dim = output_features
        n_gen = gen_input_layer = 0
        self.list_gen = []
        for _ in output_features:
            n_gen += 1
        for k, v in input_features.items():
            if k != 'noise_input':
                if isinstance(v, list) or isinstance(v, tuple):
                    gen_input_layer += math.prod(v)
                else:
                    gen_input_layer += v
                self.list_gen.append(MMgenerator(input_features=v, output_features=output_features[k],
                                                 hidden_units=hidden_units[k], n_hidden=n_hidden[k],
                                                 activ_func=activ_func))
        
        self.first_layer = MMLayer(input_features['noise_input'], hidden_units['second_layer'][0]) #[250,500]
        #self.first_layer = MMLayer(input_features['noise_input'],[45,45])
        #self.interim_layer = MMLayer([45,45],[45,45])
        #self.second_layer = MMLayer([250, 500], [370, 610])
        #self.second_layer = MMLayer([250,500],[983,250])
        self.second_layer = MMLayer(hidden_units['second_layer'][0],hidden_units['second_layer'][1])
        for n_gen in range(len(self.list_gen)):
            name = "gen" + str(n_gen)
            setattr(self,name,self.list_gen[0])
            self.list_gen.__delitem__(0)
            self.list_gen.append(self._modules["gen" + str(n_gen)])
        self.non_lin = activ_func()

    def forward(self, input, jacobian=False):
        if input.ndim == 2:
            input = input.reshape(input.shape[0], *self.noise_input)
        else:
            raise ValueError('please provide the right noise shape')
        if not jacobian:
            y = self.non_lin(self.first_layer(input))
           # y = self.non_lin(self.interim_layer(y))
            y = self.non_lin(self.second_layer(y))
            y = y.view(y.shape[0], -1) #only matrix dimensions are flattened, batch should remain same
            output = []
            index = 0
            for gn in range(len(self.list_gen)):
                out = self.list_gen[gn](y[:, index:index + math.prod(self.list_gen[gn].input_dim)].view(y.shape[0],
                                                                                    *self.list_gen[gn].input_dim))
                index += math.prod(self.list_gen[gn].input_dim)
                output.append(out)
            #print(f'index of the index {index}')
            output = tr.cat(output, dim=1)
            return output.view(output.shape[0], -1)
        else:
            y, jac = self.first_layer(input, jacobian=jacobian)
            y_1 = self.non_lin(y)
            jac = self.non_lin._jacobian_mult(x=y, val=y_1, jac_in=jac)
           # y_1_1 = self.interim_layer(y_1)
           # jac = self.interim_layer._jacobian_mult(x=y_1,val=y_1_1,jac_in=jac)
           # y_1_2 = self.non_lin(y_1_1)
           # jac = self.non_lin._jacobian_mult(x=y_1_1,val=y_1_2,jac_in=jac)
            y_2 = self.second_layer(y_1)
            jac = self.second_layer._jacobian_mult(x=y_1, val=y_2, jac_in=jac)
            y_3 = self.non_lin(y_2)
            jac = self.non_lin._jacobian_mult(x=y_2, val=y_3, jac_in=jac)
            output = []
            jac_list = []
            y_3 = y_3.view(y_3.shape[0], -1)
            jac = jac.reshape(jac.shape[0], math.prod(self.second_layer.out_features), *self.noise_input)
            index = 0
            for gn in range(len(self.list_gen)):
                out, jac_out = self.list_gen[gn](y_3[:, index: index +math.prod(self.list_gen[gn].input_dim)].view(
                    y.shape[0], *self.list_gen[gn].input_dim), jacobian=jac[:, index: index + math.prod(self.list_gen[
                   gn].input_dim), :, :].reshape(jac.shape[0], *self.list_gen[gn].input_dim, *self.noise_input))
                index += math.prod(self.list_gen[gn].input_dim)
                output.append(out)
                jac_list.append(jac_out.reshape(jac_out.shape[0], -1, math.prod(self.noise_input)))
            output = tr.cat(output, dim=1)
            return output.view(output.shape[0], -1), tr.cat(jac_list, dim=1)
        

class CorreMMGeneratorAdv(nn.Module):

    def __init__(self, input_features: dict, output_features: dict, hidden_units: dict, n_hidden: dict,
                 activ_func=nnj.ReLU, num_gen_groups: int = 6):
        super(CorreMMGeneratorAdv, self).__init__()
        try:
            self.noise_input = input_features['noise_input']
        except KeyError as e:
            raise KeyError('input_features dict does not contain noise_input key')
        self.out_dim = output_features
        n_gen = gen_input_layer = 0
        for _ in output_features:
            n_gen += 1
        self.gen_dict = dict()
        for n in range(num_gen_groups):
            list_gen = []
            for k, v in input_features.items():
                if k != 'noise_input':
                    if isinstance(v, list) or isinstance(v, tuple):
                        gen_input_layer += math.prod(v)
                    else:
                        gen_input_layer += v
                    list_gen.append(MMgenerator(input_features=v, output_features=output_features[k],
                                                     hidden_units=hidden_units[k], n_hidden=n_hidden[k],
                                                     activ_func=activ_func))
                    self.gen_dict['group' + str(n)] = list_gen
        self.first_layer = MMLayer(input_features['noise_input'], [450, 500])
        self.second_layer = MMLayer([450, 500], [600, 550])
        self.list_interim_layers = num_gen_groups*[MMLayer([600, 550], [225, 857])]

        for n_interim in range(len(self.list_interim_layers)):
            name = "interim" + str(n_interim)
            setattr(self, name, self.list_interim_layers[0])
            self.list_interim_layers.__delitem__(0)
            self.list_interim_layers.append(self._modules["interim" + str(n_interim)])

        for gen_group in self.gen_dict:
            for n_gen in range(len(self.gen_dict[gen_group])):
                name = "gen" + gen_group + '-' + str(n_gen)
                setattr(self, name, self.gen_dict[gen_group][0])
                self.gen_dict[gen_group].__delitem__(0)
                self.gen_dict[gen_group].append(self._modules["gen" + gen_group + '-' + str(n_gen)])
        self.non_lin = activ_func()

    def forward(self, input, jacobian=False):
        if input.ndim == 2:
            input = input.reshape(input.shape[0], *self.noise_input)
        else:
            raise ValueError('please provide the right noise shape')
        if not jacobian:
            y = self.non_lin(self.first_layer(input))
           # y = self.non_lin(self.interim_layer(y))
            y = self.non_lin(self.second_layer(y))
            list_y = list()
            for lyr in self.list_interim_layers:
                list_y.append(self.non_lin(lyr(y)).view(y.shape[0], -1))
            output = []
            index = 0
            for gg in self.gen_dict:
                for gn in range(len(self.gen_dict[gg])):
                    out = self.gen_dict[gg][gn](y[:, index:index + math.prod(self.gen_dict[gg][gn].input_dim)].view(y.shape[0],
                                                                                        *self.gen_dict[gg][gn].input_dim))
                    index += math.prod(self.gen_dict[gg][gn].input_dim)
                    output.append(out)
                    output = tr.cat(output, dim=1)

            return output.view(output.shape[0], -1)
        else:
            raise NotImplementedError

class BayesianMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_units: list, generator_network: Union[SimpleMLP, MMgenerator],
                 activ_func=nn.ReLU, likelihood='regression', device='cpu'):
        super(BayesianMLP, self).__init__()
        hidden_layers = len(hidden_units)  # if isinstance(hidden_units, list) else 0
        self.gen_net = generator_network.to(device)
        self.noise_dim = generator_network.input_dim
        self.non_lin = activ_func
        if isinstance(generator_network, SimpleMLP):
            w_sample = generator_network(
                tr.randn(2, self.noise_dim, device=device))  # batch norm requires multiple samples to be produced
        else:
            w_sample = generator_network(tr.randn(2, *self.noise_dim, device=device))
        self.w_dict = {'weight_sample': w_sample[0, :].squeeze(), 'param_index': 0}
        self.input_dim = input_dim
        first_layer = GenBayesianLinear(self.input_dim, hidden_units[0], weight_dict=self.w_dict, device=device)
        list_affine = [[self.non_lin(),
                        GenBayesianLinear(hidden_units[i], hidden_units[i + 1], self.w_dict, device=device)] for i in
                       range(hidden_layers - 1)]
        list_affine = [layer for lyr_active in list_affine for layer in lyr_active]
        list_affine.insert(0, first_layer)
        self.list_layers = nn.Sequential(*list_affine)  # network
        self.likelihood = likelihood
        last_layer = GenBayesianLinear(hidden_units[-1], 1, self.w_dict,
                                       device=device) if self.likelihood == 'regression' and \
                                                         hidden_units[-1] != 1 else None
        if isinstance(last_layer, GenBayesianLinear):
            self.list_layers.add_module(name='non_lin', module=self.non_lin())
            self.list_layers.add_module(name='last_layer', module=last_layer)
        print(f"No. of BNN parameters: {self.w_dict['param_index']}")

    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        assert input.shape[-1] == self.input_dim, "input dimensions not compatible with the network"
        func_out = self.list_layers(input)
        # output = nn.Softmax(func_out) if self.likelihood == 'classification' else func_out
        output = func_out
        ## use nn.NLLloss with softmax not the nn.Crossentropyloss
        return output

    # FIXME weird return signatures here
    def sample_params(self, params: Tensor = None, batch_size: int = 0, device='cpu'):
        # sample new params for the layers of the bnn
        def weights_to_layer(w_sample: Tensor) -> None:
            # assert w_sample.shape[-1] == self.gen_net.list_layers[-1].out_features, "Weight sample doesn't match " \
            #                                                                         "NN " \
            #                                                                         "params numel"
            self.w_dict['weight_sample'] = w_sample.squeeze()
            self.w_dict['param_index'] = 0
            # sets new parameters
            for l in self.list_layers:
                if isinstance(l, GenBayesianLinear):
                    l.resample_parameters(weight_dict=self.w_dict)
            # print(f"So the indices of wddeights used: {self.w_dict['param_index']}")

        if device != 'cpu':
            self.to(device)  # should shift both the bnn and the generator to gpu
        if params is None:
            if batch_size == 0:
                w_sample = self.gen_net(tr.randn(1, self.noise_dim, device=device)) if isinstance(self.gen_net,
                                                                                                  SimpleMLP) else self.gen_net(
                    tr.randn(1, *self.noise_dim, device=device))
                weights_to_layer(w_sample.squeeze())

            else:
                base_sample = tr.randn(batch_size, self.noise_dim, device=device) if isinstance(self.gen_net,
                                                                                                SimpleMLP) else tr.randn(
                    batch_size, *self.noise_dim, device=device)
                w_sample = self.gen_net(base_sample)
                return base_sample, w_sample, weights_to_layer  # this function signature is weird should really work
                # this way.

            # print(f"So the indices of weights used: {self.w_dict['param_index']}")

        else:
            assert params.shape[-1] == self.gen_net.list_layers[-1].out_features, "Weight sample doesn't match NN " \
                                                                                  "params numel"
#             if next(self.parameters()).device != 'cpu':
#                 assert params.device == next(self.parameters()).device, 'device inconsistency, make sure before calling'
            if batch_size == 0:
                assert len(params.shape) == 0 or params.shape[0] == 1, 'lot of params provided for batch size of 0'
                weights_to_layer(params)
            else:
                assert params.shape[0] == batch_size, 'parameters provided do not have the right shape, ' \
                                                      'batching impossible'
                return params, weights_to_layer


class BayesianLeNet(nn.Module):

    def __init__(self, generator_network: MMgenerator, num_classes=10, device='cpu'):
        super().__init__()
        self.gen_net = generator_network
        self.noise_dim = generator_network.input_dim
        w_sample = generator_network(tr.randn(2, *self.noise_dim, device=device))
        self.w_dict = {'weight_sample': w_sample[0, :].squeeze(), 'param_index': 0}
        self.likelihood = 'classification'
        self.list_layers = nn.Sequential(
            BayesianConv2d(1, 6, 5, weight_dict=self.w_dict),
            nn.ReLU(),
            nn.MaxPool2d(2),
            BayesianConv2d(6, 16, 5, weight_dict=self.w_dict),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            GenBayesianLinear(16 * 4 * 4, 120, self.w_dict),
            nn.ReLU(),
            GenBayesianLinear(120, 84, self.w_dict),
            nn.ReLU(),
            GenBayesianLinear(84, num_classes, self.w_dict)
        )
        print(f"No. of BNN parameters : {self.w_dict['param_index']}")

    def forward(self, x):
        return self.list_layers(x)

    def sample_params(self, params: Tensor = None, batch_size: int = 0, device='cpu'):
        # sample new params for the layers of the bnn
        def weights_to_layer(w_sample: Tensor) -> None:
            # assert w_sample.shape[-1] == self.gen_net.list_layers[-1].out_features, "Weight sample doesn't match " \
            #                                                                         "NN " \
            #                                                                         "params numel"
            self.w_dict['weight_sample'] = w_sample.squeeze()
            self.w_dict['param_index'] = 0
            # sets new parameters
            for l in self.list_layers:
                if isinstance(l, GenBayesianLinear) or isinstance(l, BayesianConv2d):
                    l.resample_parameters(weight_dict=self.w_dict)
            # print(f"So the indices of wddeights used: {self.w_dict['param_index']}")

        if device != 'cpu':
            self.to(device)  # should shift both the bnn and the generator to gpu
        if params is None:
            if batch_size == 0:
                w_sample = self.gen_net(tr.randn(1, self.noise_dim, device=device)) if isinstance(self.gen_net,
                                                                                                  SimpleMLP) else self.gen_net(
                    tr.randn(1, *self.noise_dim, device=device))
                weights_to_layer(w_sample.squeeze())

            else:
                base_sample = tr.randn(batch_size, self.noise_dim, device=device) if isinstance(self.gen_net,
                                                                                                SimpleMLP) else tr.randn(
                    batch_size, *self.noise_dim, device=device)
                w_sample = self.gen_net(base_sample)
                return base_sample, w_sample, weights_to_layer  # this function signature is weird should really work
                # this way.

            # print(f"So the indices of weights used: {self.w_dict['param_index']}")

        else:
#             assert params.shape[-1] == math.prod(self.gen_net.list_layers[-1].out_features), "Weight sample doesn't match params numel"
            if next(self.parameters()).device != 'cpu':
                assert params.device == next(self.parameters()).device, 'device inconsistency, make sure before calling'
            if batch_size == 0:
                assert len(params.shape) == 0 or params.shape[0] == 1, 'lot of params provided for batch size of 0'
                weights_to_layer(params)
            else:
                assert params.shape[0] == batch_size, 'parameters provided do not have the right shape, ' \
                                                      'batching impossible'
                return params, weights_to_layer


# only use the below for classification, general functionality to be added WIP
class PyroBNN(PyroModule):
    def __init__(self, input_dim: int, hidden_units: list, likelihood='regression', activ_func=torch.nn.ReLU):
        super(PyroBNN, self).__init__()
        assert len(hidden_units) == 2, 'nothing else supported in hidden units, len should be 3'
        assert likelihood == 'regression', 'nothing else supported but regression likelihoods'
        self.linear1 = BayesianLinear(input_dim, hidden_units[0])  # this is a PyroModule
        self.linear2 = BayesianLinear(hidden_units[0], hidden_units[1])
        # self.linear3 = BayesianLinear(hidden_units[1], hidden_units[2])
        self.non_lin = activ_func()
        to_pyro_module_(self.non_lin)
        if likelihood == 'regression':
            self.out = BayesianLinear(hidden_units[1], 1)
        self.prec = PyroSample(distppl.Gamma(9.0, 0.7))
        # self.obs_scale = 0.04 # do not sample the data noise term

    #        if torch.cuda.is_available():
    #            self.cuda()

    def forward(self, input, output=None):
        interm = self.linear1(input)  # this samples linear.bias and linear.weight
        interm = self.non_lin(interm)
        interm = self.linear2(interm)
        #         interm = self.non_lin(interm)
        #         interm = self.linear3(interm)
        interm = self.non_lin(interm)
        obs_loc = self.out(interm)
        obs_scale = 1.0 / (torch.sqrt(self.prec + 1e-8))  # this samples self.obs_scale
        with pyro.plate("batch size", input.shape[0]):
            return pyro.sample("obs", distppl.Normal(obs_loc, obs_scale).to_event(1),
                               obs=output)


class FactorisedBayesianMLP(nn.Module):
    # TODO create a factorised Bayesian MLP with independent layers, each with it's own generator, should I follow
    # KIVI's architecture to the t?
    def __init__(self, generator_list: List[MMgenerator], input_dim: int = 784,
                 activ_func=nn.ReLU, device='cpu'):
        super(FactorisedBayesianMLP, self).__init__()
        self.likelihood = 'classification'
        assert len(generator_list) == 3, 'requires multiple generator, one per each layer'
        self.gen_list = [gen.to(device) for gen in generator_list]
        self.noise_dim = generator_list[0].input_dim  # set the same noise dim for all generators
        self.non_lin = activ_func
        w_samples = [gen(tr.randn(2, *self.noise_dim, device=device)) for gen in self.gen_list]
        self.weight_dicts = list()
        for i in range(len(w_samples)):
            self.weight_dicts.append({'weight_sample': w_samples[i][0, :].squeeze(), 'param_index': 0})
        self.input_dim = input_dim
        first_layer = GenBayesianLinear(self.input_dim, 400, weight_dict=self.weight_dicts[0], device=device)
        second_layer = GenBayesianLinear(400, 400, weight_dict=self.weight_dicts[1], device=device)
        last_layer = GenBayesianLinear(400, 10, weight_dict=self.weight_dicts[2], device=device)
        print(f"Weight used in each layer {self.weight_dicts[0]['param_index']} {self.weight_dicts[1]['param_index']}"
              f" {self.weight_dicts[2]['param_index']}")
        self.list_layers = nn.Sequential(first_layer, self.non_lin(), second_layer, self.non_lin(), last_layer)

    def forward(self, input: Tensor) -> Tensor:
        input = input.reshape(input.shape[0], -1)
        assert input.shape[-1] == self.input_dim, "input dimensions not compatible with the network"
        func_out = self.list_layers(input)
        return func_out

    def sample_params(self, params: Tensor = None, batch_size: int = 0, device='cpu'):
        # sample new params for the layers of the bnn
        def weights_to_layer(w_sample: List) -> None:
            # assert w_sample.shape[-1] == self.gen_net.list_layers[-1].out_features, "Weight sample doesn't match " \
            #                                                                         "NN " \
            #                                                                         "params numel"
            for i in range(len(self.gen_list)):
                self.weight_dicts[i]['weight_sample'] = w_sample[i].squeeze()
                self.weight_dicts[i]['param_index'] = 0
            # sets new parameters
            assert len(self.gen_list) == len(self.list_layers), 'each layer has a generator'
            i = 0
            for l in self.list_layers:
                if isinstance(l, GenBayesianLinear) or isinstance(l, BayesianConv2d):
                    l.resample_parameters(weight_dict=self.weight_dicts[i])
                    i += 1
            assert i == len(self.gen_list), 'weird inconsistency here'
            # print(f"So the indices of wddeights used: {self.w_dict['param_index']}")

        if device != 'cpu':
            self.to(device)  # should move all tensors including bnn and the gen to CUDA
        if params is None:
            if batch_size == 0:
                w_sample = [gen(tr.randn(1, *self.noise_dim, device=device)).squeeze() for gen in self.gen_list]
                weights_to_layer(w_sample)

            else:
                base_sample = tr.randn(batch_size, *self.noise_dim, device=device)

                w_sample = [gen(base_sample).squeeze() for gen in self.gen_list]
                return base_sample, w_sample, weights_to_layer  # this function signature is weird should really work
                # this way.

            # print(f"So the indices of weights used: {self.w_dict['param_index']}")

        else:
            RuntimeError("this does not work rn")
            # assert params.shape[-1] == self.gen_net.list_layers[-1].out_features, "Weight sample doesn't match NN " \
            #                                                                       "params numel"
            # if next(self.parameters()).device != 'cpu':
            #     assert params.device == next(self.parameters()).device, 'device inconsistency, make sure before calling'
            # if batch_size == 0:
            #     assert len(params.shape) == 0 or params.shape[0] == 1, 'lot of params provided for batch size of 0'
            #     weights_to_layer(params)
            # else:
            #     assert params.shape[0] == batch_size, 'parameters provided do not have the right shape, ' \
            #                                           'batching impossible'
            #     return params, weights_to_layer


class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        self.net = nn.Sequential(
            tr.nn.Conv2d(1, 6, 5),
            tr.nn.ReLU(),
            tr.nn.MaxPool2d(2),
            tr.nn.Conv2d(6, 16, 5),
            tr.nn.ReLU(),
            tr.nn.MaxPool2d(2),
            tr.nn.Flatten(),
            tr.nn.Linear(16 * 4 * 4, 120),
            tr.nn.ReLU(),
            tr.nn.Linear(120, 84),
            tr.nn.ReLU(),
            tr.nn.Linear(84, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, list_hidden: List[int], activ_func=nn.ReLU):
        print("Size of the latent layer is the last element in list_hidden")
        super(MLPEncoder, self).__init__()
        self.net = SimpleMLP(input_dim=input_dim, hidden_units=list_hidden[:-1], activ_func=activ_func,
                             likelihood='vae')
        self.latent_dim = list_hidden[-1]
        self.non_lin = activ_func
        self.z_mu = nn.Linear(list_hidden[-2], self.latent_dim)
        self.z_var = nn.Linear(list_hidden[-2], self.latent_dim)

    def forward(self, x):
        out = self.net(x)
        z_mu = self.z_mu(self.non_lin()(out))
        z_var = self.z_var(self.non_lin()(out))
        return z_mu, z_var


class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, list_hidden: List[int], activ_func=nn.ReLU):
        print("Size of the latent layer is the input dim and should not be in list_hidden")
        super(MLPDecoder, self).__init__()
        self.net = SimpleMLP(input_dim=input_dim, hidden_units=list_hidden[:-1], activ_func=activ_func,
                             likelihood='vae',
                             jac_req=True)
        self.non_lin = nnj.__dict__[str(activ_func()).split(sep='(')[0]]
        self.latent_dim = input_dim
        self.rec_mu = nnj.Linear(list_hidden[-2], list_hidden[-1])
        self.rec_var = nnj.Linear(list_hidden[-2], list_hidden[-1])

    def forward(self, x, jacobian: bool = False):
        if jacobian:
            out, jac = self.net(x, jacobian)
            out_a = self.non_lin()(out)
            non_lin_jac = (self.non_lin())._jacobian_mult(x=out, val=out_a, jac_in=jac)
            x_mu = self.rec_mu(out_a)
            final_jac = self.rec_mu._jacobian_mult(out_a, x_mu, jac_in=non_lin_jac)
            # x_mu = nnj.Sigmoid()(x_mu_i)
            # final_jac = nnj.Sigmoid._jacobian_mult(x_mu_i, x_mu, jac_in=final_jac)
            x_logvar = self.rec_var(self.non_lin()(out))
            return [x_mu, x_logvar], final_jac
        else:
            return self.rec_mu(self.non_lin()(self.net(x))), self.rec_var(self.non_lin()(self.net(x)))

        # this return signature allows two returns types regardless of jac req


class MLPVAE(nn.Module):
    def __init__(self, encoder: MLPEncoder, decoder: MLPDecoder, latent_size: int, var_decoder: bool = False):
        super(MLPVAE, self).__init__()
        self.en = encoder
        self.de = decoder
        self.latent_size = latent_size
        assert self.en.latent_dim == self.de.latent_dim, "latent dimensions of the networks don't match"
        self.var_decoder = var_decoder

    def resample_latent(self, mean: Tensor, log_var: Tensor, n_samples: int = 1) -> Tensor:
        sigma = tr.exp(0.5 * log_var)
        eps = tr.randn((n_samples, *mean.shape), device=mean.device)  # n_samples is going to be 0
        new_sample = mean.expand_as(eps) + eps * sigma.expand_as(eps)
        # these samples could be generated in a batch
        # loss needs an inner loop based on this.
        return new_sample

    def forward(self, x: Tensor, jacobian: bool = False, n_samples: int = 1):

        z_mu, z_var = self.en(x)
        new_sample = self.resample_latent(z_mu, z_var, n_samples=n_samples)
        # multiple samples per input

        # FIXME there's definitely going to be a shape error here with minibatches of data.
        # reshape to sample*minibatch,*data_dim and then reshape out to minibatch,sample,*data_dim
        if new_sample.shape[0] > 1:
            batch_size = new_sample.shape[1]
            new_sample = tr.reshape(new_sample, new_sample.shape[0] * new_sample.shape[1], *new_sample.shape[1:])
            if jacobian:
                [x_mu, x_var], jac = self.de(new_sample, jacobian)
                x_mu = tr.reshape(x_mu, (-1, batch_size, *x_mu.shape[1:]))
                x_var = tr.reshape(x_var, (-1, batch_size, *x_var.shape[1:]))
                # TODO fix the jac shape at the output
                # jac = tr.mean(jac, dim=0)
                x_var = tr.mean(x_var, dim=0)
                x_mu = tr.mean(x_mu, dim=0)
                return [x_var.squeeze(), x_mu.squeeze()], jac
            else:
                x_mu, x_var = self.de(new_sample, jacobian)
                x_mu = tr.reshape(x_mu, (-1, batch_size, *x_mu.shape[1:]))
                x_var = tr.reshape(x_var, (-1, batch_size, *x_var.shape[1:]))
                x_var = tr.mean(x_var, dim=0)
                x_mu = tr.mean(x_mu, dim=0)
                return x_mu.squeeze(), x_var.squeeze()
        else:
            new_samples = tr.squeeze(new_sample)
            # process the latent samples always in minibatch,*data_dim shape
            if jacobian:
                [x_mu, x_var], jac = self.de(new_samples, jacobian)
                return [x_var, x_mu], jac
            else:
                x_mu, x_var = self.de(new_samples, jacobian)
                return x_mu, x_var
            # this return signature allows two returns types regardless of jac req

    def loss_fn(self, **kwargs):
        # USE MSE as the loss function
        kld_weight = kwargs['beta']
        input = kwargs['input']
        # TODO insert try catch block for n_samples to set to 1 sample
        n_samples = kwargs['n_samples']

        # During training of this VAE, we do not require Jacobians
        z_mu, z_var = self.en(input)
        batch_size = z_mu.shape[0]
        new_sample = self.resample_latent(z_mu, z_var, n_samples)
        if new_sample.shape[0] > 1:
            new_sample = tr.reshape(new_sample, (new_sample.shape[0] * new_sample.shape[1], *new_sample.shape[2:]))
            x_mu, x_var = self.de(new_sample)
            x_mu = tr.reshape(x_mu, (-1, batch_size, *x_mu.shape[1:]))  # correct as per sample shape, batch shape,
            # event shape
            x_var = tr.reshape(x_var, (-1, batch_size, *x_var.shape[1:]))

            # mean across samples
            x_var = tr.mean(x_var, dim=0)
            x_mu = tr.mean(x_mu, dim=0)
        else:
            x_mu, x_var = self.de(new_sample.squeeze())

        # recons_loss = F.mse_loss(input, x_mu)
        recons_loss = torch.mean((x_mu - input) ** 2)
        kld_loss = -0.5 * torch.mean(1 + z_var - z_mu ** 2 - z_var.exp())
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KL loss': -kld_loss.detach()}


class MLPVAE2(MLPVAE):
    def __init__(self, encoder: MLPEncoder, decoder: MLPDecoder, latent_size: int, var_decoder: bool = False):
        super(MLPVAE2, self).__init__(encoder, decoder, latent_size, var_decoder)
        self.prior = tr_dist.Normal(tr.zeros(self.latent_size), tr.ones(self.latent_size))

    def resample_latent(self, mean: Tensor, log_var: Tensor, n_samples: int = 1) -> Tensor:
        normal_z = tr_dist.Normal(mean, tr.exp(0.5 * log_var))
        return normal_z.sample(sample_shape=(n_samples,))

    def forward(self, x: Tensor, jacobian: bool = False, n_samples: int = 1) -> Tuple:
        z_mu, z_var = self.en(x)
        q = tr_dist.Normal(z_mu, tr.exp(0.5 * z_var))
        new_sample = q.sample(sample_shape=(n_samples,))
        if new_sample.shape[0] > 1:
            batch_size = new_sample.shape[1]
            new_sample = tr.reshape(new_sample, (new_sample.shape[0] * new_sample.shape[1], *new_sample.shape[2:]))
            if jacobian:
                [x_mu, x_var], jac = self.de(new_sample)
                x_mu = tr.reshape(x_mu, (-1, batch_size, *x_mu.shape[1:]))
                x_var = tr.reshape(x_var, (-1, batch_size, *x_var.shape[1:]))
                # return jac across all samples check entropy_acc
                return [x_mu, x_var], jac

            else:
                x_mu, x_var = self.de(new_sample)
                x_mu = tr.reshape(x_mu, (-1, batch_size, x_mu[1:]))
                x_var = tr.reshape(x_mu, (-1, batch_size, x_mu[1:]))
                return x_mu, x_var



        else:
            new_sample = new_sample.squeeze()
            if jacobian:
                [x_mu, x_var], jac = self.de(new_sample, jacobian=True)
                p = tr_dist.Normal(x_mu, tr.exp(0.5 * x_var))
                return [x_mu, x_var], jac
            else:
                x_mu, x_var = self.de(new_sample)
                p = tr_dist.Normal(x_mu, tr.exp(0.5 * x_var))
                return x_mu, x_var

    def loss_fn(self, **kwargs):
        kld_weight = kwargs['beta']
        input = kwargs['input']
        # TODO insert try catch block for n_samples to set to 1 sample
        n_samples = kwargs['n_samples'] if 'n_samples' in kwargs.keys() else 1
        analytical_kl = kwargs['kl'] if 'kl' in kwargs.keys() else False

        z_mu, z_var = self.en(input)
        batch_size = z_mu.shape[0]
        new_sample = self.resample_latent(z_mu, z_var, n_samples)
        new_sample_de = tr.reshape(new_sample, (new_sample.shape[0] * new_sample.shape[1], *new_sample.shape[2:]))
        x_mu, x_var = self.de(new_sample_de)
        # x_mu.shape[0] = sample_shape * batch_shape, x_mu.shape[1] = event_shape
        x_mu = tr.reshape(x_mu, (-1, batch_size, *x_mu.shape[1:]))
        x_var = tr.reshape(x_var, (-1, batch_size, *x_var.shape[1:]))
        # evaluate the log probabilities of the input data given and average across sample dimension
        p = tr_dist.Normal(loc=x_mu, scale=0.1)
        recons_prob = tr.sum(tr.mean(p.log_prob(input), dim=0), dim=-1)
        # for the second term of the ELBO MC approximate the KL divergence
        q = tr_dist.Normal(loc=z_mu, scale=tr.exp(0.5 * z_var))
        if analytical_kl:
            pass
        else:
            KL = tr.sum(tr.mean(q.log_prob(new_sample) - self.prior.log_prob(new_sample), dim=0), dim=-1)
        elbo = recons_prob - kld_weight * KL
        batch_elbo = tr.sum(elbo, dim=0)
        return {'loss': -batch_elbo, 'reconstruction loss': -recons_prob, 'KL loss': KL}


##########################################################################################
# Wide ResNet (for WRN16-4)
##########################################################################################
# Taken from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/models/wrn.py

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class BayesianBasicBlock(BasicBlock):
    def __init__(self, in_planes, out_planes, stride, weight_dict: dict, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes,affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = BayesianConv2d(in_planes, out_planes, kernel_size=3, weight_dict=weight_dict, stride=stride,
                                    padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes,affine=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = BayesianConv2d(out_planes, out_planes, kernel_size=3, weight_dict=weight_dict, stride=1,
                                    padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and BayesianConv2d(in_planes, out_planes, kernel_size=1,
                                                                     weight_dict=weight_dict, stride=stride,
                                                                     padding=0, bias=False) or None


class BayesianNetworkBlock(NetworkBlock):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, weight_dict: dict, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, weight_dict, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, weight_dict: dict, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, weight_dict,
                                dropRate))
        return nn.Sequential(*layers)


class BayesianWideResNet(WideResNet):
    def __init__(self, generator_network: Union[CorreMMGenerator, List[CorreMMGenerator]], depth:int,
                 widen_factor:int, num_classes=10,
                 dropRate=0.0,
                 device='cuda:0'):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BayesianBasicBlock
        if isinstance(generator_network, list):
            self.gen_net = generator_network
            list_noise_input = []
            for gen in generator_network:
                list_noise_input.append(math.prod(gen.noise_input))
                for net in gen.list_gen:
                    net.to(device)
            self.noise_dim = sum(list_noise_input)
            w_sample = tr.cat([gen(tr.randn((1, math.prod(gen.noise_input)), device=device)) for gen in
                               self.gen_net], dim=1)
        else:
            self.gen_net = generator_network.to(device)
            # for net in generator_network.list_gen:
            #     net.to(device)
            self.noise_dim = math.prod(generator_network.noise_input) #use only CorreMMGenerator
            w_sample = self.gen_net(tr.randn((1, self.noise_dim), device=device))

        self.w_dict = {'weight_sample': w_sample[0, :].squeeze(), 'param_index': 0}
        self.likelihood = 'classification'

        # 1st conv before any network block
        self.conv1 = BayesianConv2d(3, nChannels[0], kernel_size=3, weight_dict=self.w_dict, stride=1,
                                    padding=1, bias=False)
        # 1st block
        self.block1 = BayesianNetworkBlock(n, nChannels[0], nChannels[1], block, 1, self.w_dict, dropRate)
        # 2nd block
        self.block2 = BayesianNetworkBlock(n, nChannels[1], nChannels[2], block, 2, self.w_dict, dropRate)
        # 3rd block
        self.block3 = BayesianNetworkBlock(n, nChannels[2], nChannels[3], block, 2, self.w_dict, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3],affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc = GenBayesianLinear(nChannels[3], num_classes, self.w_dict)
        self.nChannels = nChannels[3]
        print(f"No. of BNN parameters: {self.w_dict['param_index']}")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, BayesianConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            #elif isinstance(m, nn.BatchNorm2d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, GenBayesianLinear):
                m.bias.data.zero_()

    def sample_params(self, params: Tensor = None, batch_size: int = 0, device='cpu'):
        # sample new params for the layers of the bnn
        def weights_to_layer(w_sample: Tensor) -> None:
            # assert w_sample.shape[-1] == self.gen_net.list_layers[-1].out_features, "Weight sample doesn't match " \
            #                                                                         "NN " \
            #                                                                         "params numel"
            self.w_dict['weight_sample'] = w_sample.squeeze()
            self.w_dict['param_index'] = 0
            # sets new parameters
            for l in self.modules():
                if isinstance(l, GenBayesianLinear) or isinstance(l, BayesianConv2d):
                    l.resample_parameters(weight_dict=self.w_dict)
            # print(f"So the indices of wddeights used: {self.w_dict['param_index']}")

        if device != 'cpu':
            self.to(device)  # should shift both the bnn and the generator to gpu
        if params is None:
            if batch_size == 0:
                if isinstance(self.gen_net, list):
                    w_sample = tr.cat([gen(tr.randn((1, math.prod(gen.noise_input)), device=device)) for gen in
                                       self.gen_net], dim=1)
                else:
                    w_sample = self.gen_net(tr.randn(1, self.noise_dim, device=device))
                weights_to_layer(w_sample.squeeze())

            else:
                if isinstance(self.gen_net, list):
                    base_sample_list = [tr.randn((batch_size, math.prod(gen.noise_input)), device=device) for gen in
                                        self.gen_net]
                    w_sample = tr.cat([gen(base_sample) for gen, base_sample in zip(self.gen_net, base_sample_list)],
                                      dim=1)
                    #base_sample = tr.cat(base_sample_list, dim=1)
                    return base_sample_list, w_sample, weights_to_layer
                else:
                    base_sample = tr.randn(batch_size, self.noise_dim, device=device)
                    w_sample = self.gen_net(base_sample)
                    return base_sample, w_sample, weights_to_layer  # this function signature is weird should really work
                # this way.

            # print(f"So the indices of weights used: {self.w_dict['param_index']}")

        else:
            assert params.shape[-1] == math.prod(self.gen_net.out_features), "Weight sample doesn't match NN " \
                                                                                  "params numel"
            if next(self.parameters()).device != 'cpu':
                assert params.device == next(self.parameters()).device, 'device inconsistency, make sure before calling'
            if batch_size == 0:
                assert len(params.shape) == 0 or params.shape[0] == 1, 'lot of params provided for batch size of 0'
                weights_to_layer(params)
            else:
                assert params.shape[0] == batch_size, 'parameters provided do not have the right shape, ' \
                                                      'batching impossible'
                return params, weights_to_layer

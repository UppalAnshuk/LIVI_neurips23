import torch
import torch as tr
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, List, Optional
from torch.nn import init
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair
from torch.nn.parameter import Parameter, UninitializedParameter
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as distppl
import math
import torch.nn.functional as F
from stochman.nnj import identity
from stochman import nnj


class GenBayesianLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    # TODO weight_sample and param_index should now be a weight_dict
    def __init__(self, in_features: int, out_features: int, weight_dict: dict, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        dtype = torch.cuda.FloatTensor if dtype is None else dtype  # make sure samples are same dtypes else redundant
        # copies
        super(GenBayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        start = weight_dict['param_index']
        end = weight_dict['param_index'] + (in_features * out_features)
        self.weight_sample = weight_dict['weight_sample'][start:end]
        # for registering I have to use a Parameter
        self.weight = self.weight_sample.view(out_features, in_features).to(device).type(dtype=dtype)
        weight_dict['param_index'] = end
        if bias:
            bias = weight_dict['weight_sample'][end:end + out_features]
            self.bias = bias.view(out_features).to(device).type(dtype=dtype)
            weight_dict['param_index'] = weight_dict['param_index'] + out_features
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()
        # TODO: updating the param index?

    def resample_parameters(self, weight_dict: dict) -> None:
        # TODO:  resampling the layer overrides method in Linear

        # # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # # https://github.com/pytorch/pytorch/issues/57109
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     init.uniform_(self.bias, -bound, bound)
        device = self.weight.device
        dtype = self.weight.dtype
        start = weight_dict['param_index']
        end = weight_dict['param_index'] + (self.in_features * self.out_features)
        self.weight_sample = weight_dict['weight_sample'][start:end]

        # for registering, have to use a Parameter
        self.weight = self.weight_sample.view(self.out_features, self.in_features).to(device).type(
            dtype=dtype)  # TODO param dict transfer from ram to vram, inefficiency?
        weight_dict['param_index'] = end
        if self.bias is not None:
            bias = weight_dict['weight_sample'][end: end + self.out_features]
            self.bias = bias.view(self.out_features).to(device).type(
                dtype=dtype)
            weight_dict['param_index'] = weight_dict['param_index'] + self.out_features

    def forward(self, inpt: Tensor) -> Tensor:
        return F.linear(inpt, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class BayesianLinear(PyroModule):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.bias = PyroSample(
            prior=distppl.Normal(0, 1.5).expand([out_size]).to_event(1))
        self.weight = PyroSample(
            prior=distppl.Normal(0, 1.5).expand([out_size, in_size]).to_event(2))

    def forward(self, input):
        return self.bias[None, :].expand(
            (input.shape[0], -1)) + input @ self.weight.T  # this line samples bias and weight


class MMLayer(nnj.AbstractJacobian, nn.Module):
    # implementation of matrix multiplication network with nnj style forward jac computation
    __constants__ = ['in_features', 'out_features']
    in_features: Union[List[int], Tuple[int]]
    out_features: Union[List[int], Tuple[int]]
    weight_l: Tensor
    weight_r: Tensor

    def __init__(self, in_features: Union[List[int], Tuple[int]], out_features: Union[List[int], Tuple[int]],
                 bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MMLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_l = Parameter(torch.empty((out_features[0], in_features[0]), **factory_kwargs))
        self.weight_r = Parameter(torch.empty((in_features[1], out_features[1]), **factory_kwargs))
        if bias:
            self.bias_l = Parameter(torch.empty((out_features[0], in_features[1]), **factory_kwargs))
            self.bias_r = Parameter(torch.empty((out_features[0], out_features[1]), **factory_kwargs))
        else:
            self.register_parameter('bias', None)  # only req for none bias
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight_l, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_r, a=math.sqrt(5))
        if self.bias_l is not None:
            fan_in_l, _ = init._calculate_fan_in_and_fan_out(self.weight_l)
            bound_l = 1 / math.sqrt(fan_in_l) if fan_in_l > 0 else 0
            init.uniform_(self.bias_l, -bound_l, bound_l)

            fan_in_r, _ = init._calculate_fan_in_and_fan_out(self.weight_r)
            bound_r = 1 / math.sqrt(fan_in_r) if fan_in_r > 0 else 0
            init.uniform_(self.bias_r, -bound_r, bound_r)

    def _left_multiplication(self, input: Tensor) -> Tensor:
        return F.linear(input.movedim(1, -1), self.weight_l, self.bias_l.T)
        # this returns transpose of the actual output of left multiplication

    def _right_multiplication(self, input: Tensor) -> Tensor:
        return F.linear(input.movedim(1, -1), self.weight_r.T, self.bias_r)

    def forward(self, input: Tensor, jacobian:Union[bool,Tensor]=False) -> Union[Tensor, Tuple[Tensor]]:
        if isinstance(jacobian, bool):
            if not jacobian:
                return self._right_multiplication(self._left_multiplication(input))  # forward op according to KIVI(
                # https://arxiv.org/abs/1705.10119)
            else:
                val = self._right_multiplication(self._left_multiplication(input))
                jac = self._jacobian_mult(x=input, val=val, jac_in=identity(input))
                return val, jac
        if isinstance(jacobian, Tensor):
            assert identity(input).shape == jacobian.shape, 'shape mismatch for jac in forward'
            val = self._right_multiplication(self._left_multiplication(input))
            jac = self._jacobian_mult(x=input, val=val, jac_in=jacobian)
            return val, jac

    def _jacobian_mult_l(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.movedim(1, -1), self.weight_l, bias=None)

    def _jacobian_mult_r(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        return F.linear(jac_in.swapaxes(1, -1), self.weight_r.T, bias=None).movedim(-1, 2)

    def _jacobian_mult(self, x: Tensor, val: Tensor, jac_in: Tensor) -> Tensor:
        #FIXME works right, but syntatically wrong as val is not the output for both
        return self._jacobian_mult_r(x, val, jac_in=self._jacobian_mult_l(x, val, jac_in))


class BayesianConv2d(nn.Conv2d):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    __annotations__ = {'bias': Optional[torch.Tensor]}

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 weight_dict: dict,  # use for resampling params
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 transposed: bool = False,
                 output_padding: Tuple[int, ...] = _pair(0),
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        dtype = torch.cuda.FloatTensor if dtype is None else dtype
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(nn.modules.conv._ConvNd, self).__init__()  # TODO check this init, is this correct usage
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size_
        self.stride = stride_
        self.padding = padding_
        self.dilation = dilation_
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        assert groups == 1, 'This layer only works for this basic functionality'
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
            if padding_ == 'same':
                for d, k, i in zip(dilation_, kernel_size_,
                                   range(len(kernel_size_) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                            total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        start = weight_dict['param_index']
        end = weight_dict['param_index'] + math.prod(self.kernel_size) * self.out_channels * self.in_channels
        self.weight_sample = weight_dict['weight_sample'][start:end]
        if transposed:
            self.weight = self.weight_sample.view(in_channels, out_channels // groups, *kernel_size_).to(device).type(
                dtype=dtype)
            # Parameter(torch.empty(
            # (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = self.weight_sample.view(out_channels, in_channels // groups, *kernel_size_).to(device).type(
                dtype=dtype)
            # Parameter(torch.empty(
            # (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        weight_dict['param_index'] = end
        if bias:
            bias = weight_dict['weight_sample'][end:end + out_channels]
            self.bias = bias.view(out_channels).to(device).type(dtype=dtype)
            weight_dict['param_index'] = weight_dict['param_index'] + out_channels
            # Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # self.reset_parameters()

    def resample_parameters(self, weight_dict: dict) -> None:
        device = self.weight.device
        dtype = self.weight.dtype
        start = weight_dict['param_index']
        end = weight_dict['param_index'] + math.prod(self.kernel_size) * self.out_channels * self.in_channels
        self.weight_sample = weight_dict['weight_sample'][start:end]
        if self.transposed:
            self.weight = self.weight_sample.view(self.in_channels, self.out_channels // self.groups,
                                                  *self.kernel_size).to(device).to(dtype=dtype)
        else:
            self.weight = self.weight_sample.view(self.out_channels, self.in_channels // self.groups,
                                                  *self.kernel_size).to(device).to(dtype=dtype)

        weight_dict['param_index'] = end

        if self.bias is not None:
            bias = weight_dict['weight_sample'][end: end + self.out_channels]
            self.bias = bias.view(self.out_channels).to(device).type(dtype=dtype)
            weight_dict['param_index'] = weight_dict['param_index'] + self.out_channels

import math
from typing import Union, Sequence

import torch
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence

from .functional import AutogradConvRNN, _conv_cell_helper
from .utils import _single, _pair, _triple


class ConvNdRNNBase(torch.nn.Module):
    def __init__(self,
                 mode,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 convndim=2,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(ConvNdRNNBase, self).__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.convndim = convndim

        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))

        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)

        self.groups = groups

        num_directions = 2 if bidirectional else 1

        if mode in ('LSTM', 'PeepholeLSTM'):
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = in_channels if layer == 0 else out_channels * num_directions
                w_ih = Parameter(torch.Tensor(gate_size, layer_input_size // groups, *self.kernel_size))
                w_hh = Parameter(torch.Tensor(gate_size, out_channels // groups, *self.kernel_size))

                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                if mode == 'PeepholeLSTM':
                    w_pi = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    w_pf = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    w_po = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
                    layer_params = (w_ih, w_hh, w_pi, w_pf, w_po, b_ih, b_hh)
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}',
                                   'weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}']
                else:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                    param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
                if bias:
                    param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']

                suffix = '_reverse' if direction == 1 else ''
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = (2 if is_input_packed else 3) + self.convndim
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                'input must have {} dimensions, got {}'.format(
                    expected_input_dim, input.dim()))
        ch_dim = 1 if is_input_packed else 2
        if self.in_channels != input.size(ch_dim):
            raise RuntimeError(
                'input.size({}) must be equal to in_channels . Expected {}, got {}'.format(
                    ch_dim, self.in_channels, input.size(ch_dim)))

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (self.num_layers * num_directions,
                                mini_batch, self.out_channels) + input.shape[ch_dim + 1:]

        def check_hidden_size(hx, expected_hidden_size, msg='Expected hidden size {}, got {}'):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode in ('LSTM', 'PeepholeLSTM'):
            check_hidden_size(hidden[0], expected_hidden_size,
                              'Expected hidden[0] size {}, got {}')
            check_hidden_size(hidden[1], expected_hidden_size,
                              'Expected hidden[1] size {}, got {}')
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            insize = input.shape[2:]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            insize = input.shape[3:]

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(self.num_layers * num_directions, max_batch_size, self.out_channels,
                                 *insize, requires_grad=False)
            if self.mode in ('LSTM', 'PeepholeLSTM'):
                hx = (hx, hx)

        self.check_forward_args(input, hx, batch_sizes)
        func = AutogradConvRNN(
            self.mode,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            variable_length=batch_sizes is not None,
            convndim=self.convndim,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups
            )
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        if self.dropout != 0:
            s += ', dropout={dropout}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(ConvNdRNNBase, self).__setstate__(d)
        if 'all_weights' in d:
            self._all_weights = d['all_weights']
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = '_reverse' if direction == 1 else ''
                if self.mode == 'PeepholeLSTM':
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}',
                               'weight_pi_l{}{}', 'weight_pf_l{}{}', 'weight_po_l{}{}',
                               'bias_ih_l{}{}', 'bias_hh_l{}{}']
                else:
                    weights = ['weight_ih_l{}{}', 'weight_hh_l{}{}',
                               'bias_ih_l{}{}', 'bias_hh_l{}{}']
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[:len(weights) // 2]]

    @property
    def all_weights(self):
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]



class Conv2dRNN(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv2dLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv2dPeepholeLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv2dGRU(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super(Conv2dGRU, self).__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dRNN(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dPeepholeLSTM(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class Conv3dGRU(ConvNdRNNBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_layers=1,
                 bias=True,
                 batch_first=False,
                 dropout=0.,
                 bidirectional=False,
                 stride=1,
                 dilation=1,
                 groups=1):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups)


class ConvRNNCellBase(torch.nn.Module):
    def __init__(self,
                 mode,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 convndim=2,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.convndim = convndim

        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))

        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)

        self.groups = groups

        if mode in ('LSTM', 'PeepholeLSTM'):
            gate_size = 4 * out_channels
        elif mode == 'GRU':
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels

        self.weight_ih = Parameter(torch.Tensor(gate_size, in_channels // groups, *self.kernel_size))
        self.weight_hh = Parameter(torch.Tensor(gate_size, out_channels // groups, *self.kernel_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(gate_size))
            self.bias_hh = Parameter(torch.Tensor(gate_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        if mode == 'PeepholeLSTM':
            self.weight_pi = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
            self.weight_pf = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))
            self.weight_po = Parameter(torch.Tensor(out_channels, out_channels // groups, *self.kernel_size))

        self.reset_parameters()

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.bidirectional is not False:
            s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.in_channels:
            raise RuntimeError(
                "input has inconsistent channels: got {}, expected {}".format(
                    input.size(1), self.in_channels))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.out_channels:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.out_channels))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        self.check_forward_input(input)

        if hx is None:
            batch_size = input.size(0)
            insize = input.shape[2:]
            hx = input.new_zeros(batch_size, self.out_channels, *insize, requires_grad=False)
            if self.mode in ('LSTM', 'PeepholeLSTM'):
                hx = (hx, hx)
        if self.mode in ('LSTM', 'PeepholeLSTM'):
            self.check_forward_hidden(input, hx[0])
            self.check_forward_hidden(input, hx[1])
        else:
            self.check_forward_hidden(input, hx)

        cell = _conv_cell_helper(
            self.mode,
            convndim=self.convndim,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups)
        if self.mode == 'PeepholeLSTM':
            return cell(
                input, hx,
                self.weight_ih, self.weight_hh, self.weight_pi, self.weight_pf, self.weight_po,
                self.bias_ih, self.bias_hh
            )
        else:
            return cell(
                input, hx,
                self.weight_ih, self.weight_hh,
                self.bias_ih, self.bias_hh,
            )


class Conv1dRNNCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv1dLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv1dPeepholeLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv1dGRUCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=1,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dRNNCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dPeepholeLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv2dGRUCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dRNNCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 nonlinearity='tanh',
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        if nonlinearity == 'tanh':
            mode = 'RNN_TANH'
        elif nonlinearity == 'relu':
            mode = 'RNN_RELU'
        else:
            raise ValueError("Unknown nonlinearity '{}'".format(nonlinearity))
        super().__init__(
            mode=mode,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='LSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dPeepholeLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='PeepholeLSTM',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )


class Conv3dGRUCell(ConvRNNCellBase):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 dilation=1,
                 groups=1
                 ):
        super().__init__(
            mode='GRU',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=bias,
            convndim=3,
            stride=stride,
            dilation=dilation,
            groups=groups
        )

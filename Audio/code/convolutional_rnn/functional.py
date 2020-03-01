from functools import partial

import torch
import torch.nn.functional as F
#from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend

from .utils import _single, _pair, _triple


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    hy = F.relu(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    hy = F.tanh(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear:
        igates = linear_func(input, w_ih)
        hgates = linear_func(hidden[0], w_hh)
        #state = fusedBackend.LSTMFused.apply
        #return state(igates, hgates, hidden[1]) if b_ih is None else state(igates, hgates, hidden[1], b_ih, b_hh)
        # Slice off the workspace arg (used only for backward)
        return _cuda_fused_lstm_cell(igates, hgates, hidden[1], b_ih, b_hh)[:2]

    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy


def PeepholeLSTMCell(input, hidden, w_ih, w_hh, w_pi, w_pf, w_po,
                     b_ih=None, b_hh=None, linear_func=None):
    if linear_func is None:
        linear_func = F.linear
    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate += linear_func(cx, w_pi)
    forgetgate += linear_func(cx, w_pf)
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    outgate += linear_func(cy, w_po)
    outgate = F.sigmoid(outgate)

    hy = outgate * F.tanh(cy)

    return hy, cy


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """ Copied from torch.nn._functions.rnn and modified """
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear:
        gi = linear_func(input, w_ih)
        gh = linear_func(hidden, w_hh)
        #state = fusedBackend.GRUFused.apply
        #return state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)
        return _cuda_fused_gru_cell(gi, gh, hidden, b_ih, b_hh)[0]
    gi = linear_func(input, w_ih, b_ih)
    gh = linear_func(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):
    """ Copied from torch.nn._functions.rnn and modified """

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        assert(len(weight) == total_layers)
        next_hidden = []
        ch_dim = input.dim() - weight[0][0].dim() + 1

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, ch_dim)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size())
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size())

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    """ Copied from torch.nn._functions.rnn without any modification """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(inner, reverse=False):
    """ Copied from torch.nn._functions.rnn without any modification """
    if reverse:
        return VariableRecurrentReverse(inner)
    else:
        return VariableRecurrent(inner)


def VariableRecurrent(inner):
    """ Copied from torch.nn._functions.rnn without any modification """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset:input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(inner):
    """ Copied from torch.nn._functions.rnn without any modification """
    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[:batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:batch_size]), 0)
                               for h, ih in zip(hidden, initial_hidden))
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size:input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def ConvNdWithSamePadding(convndim=2, stride=1, dilation=1, groups=1):
    def forward(input, w, b=None):
        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError('convndim must be 1, 2, or 3, but got {}'.format(convndim))

        if input.dim() != convndim + 2:
            raise RuntimeError('Input dim must be {}, bot got {}'.format(convndim + 2, input.dim()))
        if w.dim() != convndim + 2:
            raise RuntimeError('w must be {}, bot got {}'.format(convndim + 2, w.dim()))

        insize = input.shape[2:]
        kernel_size = w.shape[2:]
        _stride = ntuple(stride)
        _dilation = ntuple(dilation)

        ps = [(i + 1 - h + s * (h - 1) + d * (k - 1)) // 2
              for h, k, s, d in list(zip(insize, kernel_size, _stride, _dilation))[::-1] for i in range(2)]
        # Padding to make the output shape to have the same shape as the input
        input = F.pad(input, ps, 'constant', 0)
        return getattr(F, 'conv{}d'.format(convndim))(
            input, w, b, stride=_stride, padding=ntuple(0), dilation=_dilation, groups=groups)
    return forward


def _conv_cell_helper(mode, convndim=2, stride=1, dilation=1, groups=1):
    linear_func = ConvNdWithSamePadding(convndim=convndim, stride=stride, dilation=dilation, groups=groups)

    if mode == 'RNN_RELU':
        cell = partial(RNNReLUCell, linear_func=linear_func)
    elif mode == 'RNN_TANH':
        cell = partial(RNNTanhCell, linear_func=linear_func)
    elif mode == 'LSTM':
        cell = partial(LSTMCell, linear_func=linear_func)
    elif mode == 'GRU':
        cell = partial(GRUCell, linear_func=linear_func)
    elif mode == 'PeepholeLSTM':
        cell = partial(PeepholeLSTMCell, linear_func=linear_func)
    else:
        raise Exception('Unknown mode: {}'.format(mode))
    return cell


def AutogradConvRNN(
        mode, num_layers=1, batch_first=False,
        dropout=0, train=True, bidirectional=False, variable_length=False,
        convndim=2, stride=1, dilation=1, groups=1):
    """ Copied from torch.nn._functions.rnn and modified """
    cell = _conv_cell_helper(mode, convndim=convndim, stride=stride, dilation=dilation, groups=groups)

    rec_factory = variable_recurrent_factory if variable_length else Recurrent

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(layer, num_layers, (mode in ('LSTM', 'PeepholeLSTM')), dropout=dropout, train=train)

    def forward(input, weight, hidden, batch_sizes):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight, batch_sizes)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward

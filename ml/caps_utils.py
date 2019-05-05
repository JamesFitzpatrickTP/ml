import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np


def axis_string(max_dims):
    strings = (chr(97 + i) for i in range(max_dims))
    string = ''.join(strings)
    return string


def index_string(string, indices):
    string = np.array(list(string))
    string = string[indices]
    return ''.join(string)


def ein_string(string, substring):
    indices = [(substring[i] in string) for i in range(len(substring))]
    indices = ~ np.array(indices).astype(bool)
    indices = list(indices) + [True] * int(len(string) - len(substring))
    indices = np.array(indices)
    return index_string(string, indices)


def parse_axes(string, axes):
    np_axes = np.array(axes)
    string_a = index_string(string, np_axes[:, 0])
    string_b = index_string(string, np_axes[:, 1])
    string_a = ein_string(string, string_a)
    string_b = ein_string(string, string_b)
    return string_a + string_b


def tensordot(tensor, weights, axes):
    max_dims = max(tensor.dim(), weights.dim())
    string = axis_string(max_dims)
    string_c = ''.join(sorted(set(parse_axes(string, axes))))
    string_a, string_b = string[:tensor.dim()], string[:weights.dim()]
    string = string_a + ',' + string_b + '->' + string_c
    return torch.einsum(string, (tensor, weights))


def compute_padding(in_dim, out_dim, ker_size, stride):
    numerator = ker_size - in_dim + stride * (out_dim - 1)
    return int(numerator / 2)


def set_padding(in_dim, out_dim, ker_size, stride, mode):
    if not isinstance(mode, str):
        raise TypeError("Padding must be a string of same, half or none")
    if mode == 'same':
        padding = compute_padding(in_dim, in_dim, ker_size, stride)
    elif mode == 'half':
        padding = compute_padding(in_dim, in_dim / 2, ker_size, stride) + 1
    else:
        padding = 0
    return padding


def slice_tensor(tensor, i, j, ker_width):
    slices = [slice(None)] * tensor.dim()
    slices[-2] = slice(i, i + ker_width)
    slices[-1] = slice(j, j + ker_width)
    return(tensor[slices])
    


def strided(tensor, window_shape, stride=None, axis=None):
    shape = torch.tensor(tensor.shape)

    if axis is not None:
        axs = torch.tensor(axis)
    else:
        axs = torch.arange(len(shape))

    window_shape = torch.tensor(window_shape)
    window = shape + 0
    window[axs] = window_shape

    ones = torch.ones_like(shape)
    if stride:
        stride = torch.tensor(stride)
        ones[axs] = stride
        ones[0], ones[1] = 1, 1

    strides = torch.tensor(tensor.stride())

    shape = tuple((shape - window) // ones + 1) + tuple(window)
    strides = tuple(strides * ones) + tuple(strides)
    return tensor.as_strided(shape, strides)


def tensor_convND(tensor, weights, ker_width, stride=None, padding='same'):
    padding = set_padding(tensor.shape[3], tensor.shape[3], ker_width, stride, padding)
    pad = torch.nn.ZeroPad2d(padding)
    new_tensor = pad(tensor)
    new_tensor = strided(new_tensor, (1, 1, ker_width, ker_width), stride=stride)
    new_tensor = new_tensor.squeeze(dim=-3)
    new_tensor = new_tensor.squeeze(dim=-3)
    tensor_product = torch.einsum('abcdef,efbhi->acdhi', (new_tensor, weights))
    return tensor_product

import torch
import numpy as np


def standard(tensor, axis=0):
    mean = tensor.mean(dim=axis)
    div = np.prod([tensor.shape[idx] for idx in tuple(axis)])
    var = ((tensor - mean) ** 2).sum(dim=axis) / div
    std = (var) ** 0.5
    print(var, div, std)
    return std


def zero_mean_unit_variance(tensor, axis=0):
    mean = tensor.mean(dim=axis)
    std = standard(tensor, axis=axis)
    return (tensor - mean) / std


def clip(tensor, limits=None, axis=0):
    if limits is not None:
        tensor = torch.clamp(tensor, *limits)
    lwr, upr = tensor.min(dim=axis), tensor.max(dim=axis)
    tensor = (tensor - lwr) / (upr - lwr)
    return tensor


def clip_between(tensor, limits=None, clip_range=None, axis=0):
    tensor = clip(tensor=tensor, limits=limits, axis=axis)
    if clip_range is not None:
        tensor = tensor - clip_range[0]
        tensor = tensor * (clip_range[1] - clip_range[0])
    return tensor

import numpy as np
import torch


def to_tensor(arr):
    return torch.tensor(arr).float()


def expand_arr(arr):
    arr = np.expand_dims(arr, 0)
    return np.expand_dims(arr, 0)


def check_window(shape, window):
    assert len(shape) == len(window), 'shape and window must have same length'
    assert np.all(np.array(shape) - np.array(window) >= 0), 'window must be smaller' 
    

def gen_crop(shape, window):
    check_window(shape, window)
    lims = np.array(shape) - np.array(window)
    srts = [np.random.randint(lim + 1) for lim in lims]
    slices = [slice(srt, srt + win) for srt, win in zip(srts, window)]
    return slices


def gen_crop_multiplet(shape, window, num=2):
    crop = gen_crop(shape, window)
    return (crop, ) * num


def crop_multiplet(multiplet, crops):
    return [item[crop] for item, crop in zip(multiplet, crops)]

import matplotlib.pyplot as plt
import pydicom as pd
import numpy as np
import os


def blank(*args):
    return args


def list_files(path):
    return os.listdir(path)


def list_fnames(path):
    files = list_files(path)
    return [os.path.join(path, file) for file in files]


def join_fnames(dir_name, fnames):
    return [os.path.join(dir_name, fname) for fname in fnames]


def list_fnames_recursive(path):
    fnames = [join_fnames(triplet[0], triplet[-1])
              for triplet in os.walk(path)]
    return [item for fname in fnames for item in fname]


def pair_training_set(sample_fnames, target_fnames, fun=None, sorter=None):
    if fun is None:
        fun = blank
    if sorter is None:
        sorter = blank

    sample_fnames, target_fnames = fun(sorter(sample_fnames, target_fnames))
    return zip(sample_fnames, target_fnames)


def load_img(path):
    return plt.imread(path)


def load_dcm(path):
    return pd.read_file(path)


def load_dcm_arr(path):
    return load_dcm(path).pixel_array


def load_imgs(iterable, fun=None):
    if fun is None:
        fun = blank
        
    return fun(load_img(item) for item in iterable)


def load_dcms(iterable, fun=None):
    if fun is None:
        fun = blank
        
    return fun(load_dcm(item) for item in iterable)


def load_dcm_arrs(iterable, fun=None):
    if fun is None:
        fun = blank
        
    return fun([load_dcm_arr(item) for item in iterable])

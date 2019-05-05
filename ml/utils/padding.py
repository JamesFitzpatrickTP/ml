import numpy as np


def find_next_factor(size, divisor):
    """
    Given an image edge length, compute
    the next factor of the divisor it needs
    to be to fit in an upsampling / downsamling
    network
    """
    ratio = size / divisor
    is_okay = np.ceil(ratio - int(ratio)).astype(int)
    return (int(ratio) + 1 * is_okay) * divisor


def allocate_pad(size, divisor):
    increase = find_next_factor(size, divisor) - size
    if increase % 2 > 0:
        return (int((increase + 1) / 2), int((increase + 1) / 2) - 1, )
    else:
        return (int(increase / 2), int(increase / 2), )


def img_padding(img, divisor):
    shape = img.shape
    padding = [allocate_pad(size, divisor) for size in shape]
    return tuple(padding)


def rgb_padding(img, divisor):
    shape = img.shape
    padding = [allocate_pad(size, divisor) for size in shape[:-1]]
    return tuple(padding)


def pad_img(img, padding):
    return np.pad(img, padding, mode='constant')


def pad_rgb(img, padding):
    return np.pad(img, padding + ((0, 0),), mode='constant')


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

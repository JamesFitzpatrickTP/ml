import torch.nn as nn


class ASPP(nn.Module):
    def __init__(self, in_maps, out_maps, in_dims, out_dims,
                 ker_sizes, dilations, strides):
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.ker_sizes = ker_sizes
        self.dilations = dilations
        self.strides = strides

        self.paddings = [pd.compute_padding(self.in_dims,
                                            self.out_dims,
                                            ker_size * (1 + (dilation - 1)),
                                            stride)
                         for ker_size, stride, dilation in zip(
                                 self.ker_sizes,
                                 self.strides,
                                 self.dilations)]

    def make_assedrtions(self):
        assert len(self.strides) == len(self.dilations)
        assert len(self.strides) == len(self.ker_sizes)


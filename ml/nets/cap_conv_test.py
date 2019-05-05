import torch
import torch.nn as nn

from ml import layers as cl

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.convolution_one = cl.Convolve(1, 8, 5, 1, 'same')
        self.downsampling_one = cl.CapConv2D(8, 8, 5, 'half', 2, 1, 2)
        self.capsuling_one = cl.CapConv2D(8, 8, 5, 'same', 1, 3, 4)
        self.upsampling_three = cl.CapConvTranspose2D(8, 8, 5, 'same', 1, 3, 2)
        self.capsuling_nine = cl.CapConv2D(8, 8, 1, 'same', 1, 3, 2)
        self.capsuling_ten = cl.CapConv2D(8, 8, 5, 'same', 1, 3, 1)
        self.convolution_two = cl.Convolve(8, 1, 5, 1, 'same')


    def forward(self, tensor):
        a = nn.ReLU()(self.convolution_one(tensor))
        b = self.downsampling_one(a)
        c = self.capsuling_one(b)
        o = self.upsampling_three(c)
        p = self.capsuling_nine(torch.cat((a, o)))
        q = self.capsuling_ten(p)
        r = self.convolution_two(q)
        return nn.Sigmoid()(torch.squeeze(r))

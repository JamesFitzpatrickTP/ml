import torch
import torch.nn as nn
import torch.nn.functional as fn

from ml import caps_utils


class Convolve(nn.Module):
    def __init__(self, in_maps=1, out_maps=1,
                 ker_size=3, stride=1, mode='none'):
        super(Convolve, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.stride = stride

        self.padding = caps_utils.set_padding(
            in_maps, out_maps, ker_size, stride, mode)
        
        self.conv = nn.Conv2d(
            in_channels=self.in_maps,
            out_channels=self.out_maps,
            kernel_size=self.ker_size,
            stride=self.stride,
            padding=self.padding
        )

    def forward(self, tensor):
        return fn.relu(self.conv(tensor))


class CapConv2D(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, mode,
                 stride, num_routes, num_atoms,
                 padding='same'):
        super(CapConv2D, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.mode = mode
        self.stride = stride
        self.num_routes = num_routes
        self.num_atoms = num_atoms

        self.weights = nn.Parameter(torch.randn((self.ker_size,
                                                 self.ker_size,
                                                 self.in_maps,
                                                 self.num_atoms,
                                                 self.out_maps)).float())

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, tensor):
        votes = self.voting(tensor, self.weights)
        votes = votes.permute(0, 3, 4, 1, 2)
        votes = self.routing(votes, self.num_routes)
        return votes

    def squash(self, tensor, dim=0, epsilon=1e-4):
        squared_norm = (tensor ** 2).sum(1, keepdim=True) + epsilon
        norm = torch.sqrt(squared_norm)
        numerator = squared_norm * tensor
        denominator = (1 + squared_norm) * norm
        return numerator / (denominator + epsilon)

    def voting(self, tensor, weights):
        tensor = caps_utils.tensor_convND(tensor, weights, self.ker_size,
                                          self.stride, self.mode)
        return tensor

    def routing(self, votes, num_routes=1):
        a, b, c, d, e = votes.shape
        logits = torch.zeros((a, b, d, e)).float()
        for routes in range(num_routes):
            logits = self.softmax(logits)
            preds = torch.einsum('abcde,abde->bcde', (votes, logits))
            preds = self.squash(preds, dim=1)
            logits = logits + torch.einsum('abcde,bcde->abde', (votes, preds))
        return preds


class CapConvTranspose2D(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, mode,
                 stride, num_routes, num_atoms,
                 padding='same', scale_factor=2):
        super(CapConvTranspose2D, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.mode = mode
        self.stride = stride
        self.num_routes = num_routes
        self.num_atoms = num_atoms
        self.scale_factor = scale_factor

        self.weights = torch.randn((self.ker_size,
                                    self.ker_size,
                                    self.in_maps,
                                    self.num_atoms,
                                    self.out_maps)).float()

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, tensor):
        tensor = fn.interpolate(tensor, scale_factor=self.scale_factor)
        votes = self.voting(tensor, self.weights)
        votes = votes.permute(0, 3, 4, 1, 2)
        votes = self.routing(votes, self.num_routes)
        return votes

    def squash(self, tensor, dim=0, epsilon=1e-4):
        squared_norm = (tensor ** 2).sum(1, keepdim=True) + epsilon
        norm = torch.sqrt(squared_norm)
        numerator = squared_norm * tensor
        denominator = (1 + squared_norm) * norm
        return numerator / (denominator + epsilon)

    def voting(self, tensor, weights):
        tensor = caps_utils.tensor_convND(tensor, weights, self.ker_size,
                                          self.stride, self.mode)
        return tensor

    def routing(self, votes, num_routes=1):
        a, b, c, d, e = votes.shape
        logits = torch.zeros((a, b, d, e)).float()
        for routes in range(num_routes):
            logits = self.softmax(logits)
            preds = torch.einsum('abcde,abde->bcde', (votes, logits))
            preds = self.squash(preds, dim=1)
            logits = logits + torch.einsum('abcde,bcde->abde', (votes, preds))
        return preds


class DownCaps(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, num_routes, num_atoms):
        super(DownCaps, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.num_routes = num_routes
        self.num_atoms = num_atoms

        self.down = CapConv2D(self.in_maps, self.out_maps, self.ker_size,
                              'half', 2, self.num_routes, self.num_atoms)
        self.caps = CapConv2D(self.out_maps, self.out_maps, self.ker_size,
                              'same', 1, self.num_routes, self.num_atoms)

    def forward(self, tensor):
        tensor = self.down(tensor)
        return self.caps(tensor)


class UpConcatCaps(nn.Module):
    def __init__(self, in_maps, out_maps, ker_size, num_routes, num_atoms):
        super(UpConcatCaps, self).__init__()

        self.in_maps = in_maps
        self.out_maps = out_maps
        self.ker_size = ker_size
        self.num_routes = num_routes
        self.num_atoms = num_atoms

        self.up = CapConvTranspose2D(self.in_maps, self.out_maps, self.ker_size,
                                     'same', 1, self.num_routes, int(self.num_atoms / 2), 2)
        self.caps_a = CapConv2D(self.in_maps, self.out_maps, self.ker_size,
                                'same', 1, self.num_routes, self.num_atoms)
        self.caps_b = CapConv2D(self.out_maps, self.out_maps, self.ker_size,
                                'same', 1, self.num_routes, self.num_atoms)

    def forward(self, tensor_a, tensor_b):
        tensor = self.up(tensor_a)
        tensor = self.caps_b(torch.cat((tensor, tensor_b)))
        return self.caps_b(tensor)
        

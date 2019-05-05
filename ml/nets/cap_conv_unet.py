import torch
import torch.nn as nn

from ml import layers as cl


class Network(nn.Module):
    def __init__(self, filter_ns=None, batch_size=1, in_channels=1,
                 out_channels=1, ker_size=5, route_ns=None, atom_ns=None):
        super(Network, self).__init__()

        self.filter_ns = filter_ns
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ker_size = ker_size
        self.route_ns = route_ns
        self.atom_ns = atom_ns

        self.down_funs = []
        self.up_funs = []
        
        self.check_filters()
        self.check_routes()
        self.check_atoms()

        self.in_fun = cl.Convolve(self.in_channels, self.filter_ns[0],
                                  self.ker_size, mode='same')
        
        for in_maps, out_maps, routes, atoms in zip([self.filter_ns[0]] + self.filter_ns,
                                                    self.filter_ns,
                                                    self.route_ns, self.atom_ns):
            fun = cl.DownCaps(in_maps, out_maps, self.ker_size, routes, atoms)
            self.down_funs.append(fun)

        rev_a = reversed(self.filter_ns)
        rev_b = reversed([self.filter_ns[0]] + self.filter_ns[:-1])
        rev_c = reversed(self.route_ns)
        rev_d = reversed(self.atom_ns)

        for in_maps, out_maps, routes, atoms in zip(rev_a, rev_b, rev_c, rev_d):
            print('ATOMS: ', atoms, in_maps, out_maps)
            fun = cl.UpConcatCaps(in_maps, out_maps, self.ker_size, routes, atoms)
            self.up_funs.append(fun)

        self.out_caps = cl.CapConv2D(self.filter_ns[0], self.out_channels, self.ker_size,
                                  "same", 1, 1, 1)    
        self.out_fun = cl.Convolve(self.out_channels, self.out_channels,
                                   1, mode='same')

    def check_filters(self):
        if self.filter_ns is None:
            raise ValueError('filter_ns cannot be None')

    def check_atoms(self):
        if self.atom_ns is None:
            raise ValueError('atom_ns cannot be None')
        if self.filter_ns is not None:
            if len(self.filter_ns) != len(self.atom_ns):
                raise ValueError('there must be an equal number of filters and atoms')

    def check_routes(self):
        if self.route_ns is None:
            print('defaulting routes to three for all capsules...')
            self.route_ns = [3 for i in range(len(self.filter_ns))]

    def forward(self, tensor):
        # tensor = nn.BatchNorm2d(self.in_channels)(tensor)
        tensor = nn.ReLU()(self.in_fun(tensor))
        down_tensors = []
        down_tensors.append(tensor)
         
        for fun in self.down_funs:
            tensor = fun(tensor)
            down_tensors.append(tensor)
        
        rev_e = reversed(down_tensors[:-1])
        
        for fun, old_tens in zip(self.up_funs, rev_e):
            tensor = fun(tensor, old_tens)

        tensor = self.out_caps(tensor)
        tensor = self.out_fun(tensor)
        return nn.Sigmoid()(tensor)

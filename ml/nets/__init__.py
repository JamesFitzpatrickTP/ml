from . import cap_conv_unet
from . import cap_conv_test


nets = {
    'cap_unet': cap_conv_unet,
    'cap_test': cap_conv_test,
}


def get_net(name, **conf):
    return nets[name](**conf)

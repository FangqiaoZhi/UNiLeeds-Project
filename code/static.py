import code.model as model

import itertools
import numpy as np
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as tmf
import pandas as pd

from code.config import lp_method

# Create U-Net Network Model
# input: RGB 3 channels
# output: true / false 2 channels in softmax method
#         float value  1 channel in lp_method
_output_channel = 1 if lp_method else 2
_unet_model = model.UNet(3, _output_channel)

# Check if there is gpu
_has_gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if _has_gpu else 'cpu')
net = _unet_model.cuda() if _has_gpu else _unet_model

_test_number = 1

def get_net():
    global net
    global _test_number
    #print('_test_number', _test_number)
    return net, _test_number

def set_net(this_net, magic):
    global net
    global _test_number
    net, _test_number = this_net, magic
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F


class Nonlinearity(torch.nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        #return F.selu(x)
        #return F.relu(x)
        #return x
        return F.leaky_relu(x)
        #return x + torch.sin(10*x)/5
        #return x + torch.sin(x)
        #return x + torch.sin(x) / 2
        #return x + torch.sin(4*x) / 2
        #return torch.cos(x) - x
        #return x * F.sigmoid(x)
        #return torch.exp(x)#x**2
        #return x - .1*torch.sin(5*x)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        k = 1024
        input_size = 911
        # 911 dims for cov2 a549
        # 924 dims for cov2 calu
        # 930 dims for iav a549
        # 921 dims for rsv a549
        # 925 dims for hpiv3 a549
        self.net = nn.Sequential(nn.Linear(input_size, k, bias=False),
                                 Nonlinearity(),
                                 nn.Linear(k, input_size, bias=False))

    def forward(self, x):
        return self.net(x)

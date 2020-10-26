import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import neural_model
import numpy as np
import visdom
from copy import deepcopy
import math
import pickle as p
import random
from torchvision.utils import make_grid
import time
import torch.optim.lr_scheduler as lr_scheduler


def train_network(train_loader, test_loader):

    # Uncomment below to resume training if needed
    #d = torch.load('trained_model_best.pth')
    net = neural_model.Net()
    #net.load_state_dict(d['state_dict'])

    # Print Num of Params in Model
    params = 0
    for idx, param in enumerate(list(net.parameters())):
        size = 1
        for idx in range(len(param.size())):
            size *= param.size()[idx]
            params += size
    print("NUMBER OF PARAMS: ", params)

    # Custom Initialization if needed
    """
    bound = 1e-10
    for idx, param in enumerate(net.parameters()):
        if idx == len(list(net.parameters())) - 1:
            print(param.size())
            param.data.fill_(0)
        else:
            init = torch.Tensor(param.size()).uniform_(-bound, bound)
            param.data = init
    #"""

    # Adam optimization (but you can try SGD as well)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    #optimizer = torch.optim.SGD(net.parameters(), lr=1e-1)

    net.cuda()
    num_epochs = 100000
    best_loss = np.float("inf")

    for i in range(num_epochs):

        print("Epoch: ", i)
        train_loss = train_step(net, optimizer, train_loader)
        print("Train Loss: ", train_loss)
        test_loss = val_step(net, test_loader)
        print("Test Loss: ", test_loss)

        if train_loss < 1e-15:
            break
        if test_loss < best_loss:
            best_loss = test_loss
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            torch.save(d, 'trained_model_best.pth')
            net.cuda()
        print("Best Test Loss: ", best_loss)


def train_step(net, optimizer, train_loader):
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)
    criterion = torch.nn.MSELoss()

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()

        inputs = Variable(batch).cuda()
        output = net(inputs)

        loss = criterion(output, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)

    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, batch in enumerate(val_loader):
        inputs = Variable(batch).cuda()
        with torch.no_grad():
            output = net(inputs)
        loss = criterion(output, inputs)

        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss

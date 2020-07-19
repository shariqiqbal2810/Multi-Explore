import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain

class SmallConvNet(nn.Module):
    def __init__(self, img_dim, out_dim=32, dim_factor=32, nonlin=F.relu,
                 strides=[2, 2, 2, 2]):
        super(SmallConvNet, self).__init__()
        C, H, W = img_dim
        stride_reduction = np.prod(strides)
        conv_out_dim = (W // stride_reduction) * (H // stride_reduction) * dim_factor

        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(C, dim_factor, 3, stride=strides[0])
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim_factor, dim_factor, 3, stride=strides[1])
        self.pad3 = nn.ReflectionPad2d(1)
        self.conv3 = nn.Conv2d(dim_factor, dim_factor, 3, stride=strides[2])
        self.pad4 = nn.ReflectionPad2d(1)
        self.conv4 = nn.Conv2d(dim_factor, dim_factor, 3, stride=strides[2])
        self.fc1 = nn.Linear(conv_out_dim, out_dim)
        # self.fc2 = nn.Linear(dim_factor * 4, out_dim)

        self.nonlin = nonlin

    def forward(self, inp, norm_in=True):
        if norm_in:
            inp /= 255.0
        out1 = self.nonlin(self.conv1(self.pad1(inp)))
        out2 = self.nonlin(self.conv2(self.pad2(out1)))
        out3 = self.nonlin(self.conv3(self.pad3(out2)))
        out4 = self.nonlin(self.conv4(self.pad4(out3)))
        out4_flat = out4.view(out4.shape[0], -1)
        out5 = self.nonlin(self.fc1(out4_flat))
        # out6 = self.nonlin(self.fc2(out5))
        return out5


class CombineNet(nn.Module):
    """
    Processes and combines image and vector observations
    """
    def __init__(self, obs_dim, out_dim, hidden_dim=32, nonlin=F.relu,
                 n_heads=1):
        super(CombineNet, self).__init__()
        img_dim, vect_dim = obs_dim
        self.convnet = SmallConvNet(img_dim, out_dim=hidden_dim * 4, dim_factor=hidden_dim)
        self.vect_fc = nn.Linear(vect_dim[0], hidden_dim)
        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        if n_heads >= 1:
            # self.fc_out = nn.ModuleList([nn.Linear(hidden_dim, out_dim)
            #                              for _ in range(n_heads)])
            self.fc_out = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 5, hidden_dim),
                                                       nn.ReLU(),
                                                       nn.Linear(hidden_dim, out_dim))
                                         for _ in range(n_heads)])
        else:
            raise Exception('n_heads must be >= 1')
        self.nonlin = nonlin
        self.n_heads = n_heads
        self.shared_modules = [self.convnet, self.vect_fc] #, self.fc1]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.n_heads)

    def forward(self, inp, head=None):
        imgs, vects = inp
        conv_out = self.convnet(imgs)
        vect_out = self.nonlin(self.vect_fc(vects))
        cat_in = torch.cat((conv_out, vect_out), dim=1)
        # h1 = self.nonlin(self.fc1(cat_in))
        if head is None:
            out = [f(cat_in) for f in self.fc_out]
        elif type(head) is list:
            out = [self.fc_out[h](cat_in) for h in head]
        else:
            out = self.fc_out[head](cat_in)
        return out

class MLPNet(nn.Module):
    """
    Processes vector observations
    """
    def __init__(self, vect_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 n_heads=1):
        super(MLPNet, self).__init__()
        self.vect_fc = nn.Sequential(nn.Linear(vect_dim, hidden_dim * 4),
                                     nn.ReLU())
        if n_heads >= 1:
            self.fc_out = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim * 4,
                                                                 hidden_dim),
                                                       nn.ReLU(),
                                                       nn.Linear(hidden_dim,
                                                                 out_dim))
                                         for _ in range(n_heads)])
        else:
            raise Exception('n_heads must be >= 1')
        self.nonlin = nonlin
        self.n_heads = n_heads
        self.shared_modules = [self.vect_fc] #, self.fc1]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.n_heads)

    def forward(self, inp, head=None):
        vect_out = self.vect_fc(inp)
        # h1 = self.nonlin(self.fc1(cat_in))
        if head is None:
            out = [f(vect_out) for f in self.fc_out]
        elif type(head) is list:
            out = [self.fc_out[h](vect_out) for h in head]
        else:
            out = self.fc_out[head](vect_out)
        return out

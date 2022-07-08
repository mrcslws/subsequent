"""
This is an initial naive version of the feedforward algorithm. See the
TensorFlow version for the latest algorithm. This version works only on MNIST;
it has not yet been adapted to work on images with more than one channel. It
also uses a naive form of hypernetwork. The TensorFlow version is more advanced,
using more weight-sharing within the hypernetwork.

This code at least serves as a jumping-off point. It shows how to take a batch
of dynamic weights and run it efficiently on the respective images using a group
convolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFunction(nn.Module):
    def __init__(self, tensors):
        super().__init__()

        (self.weight1,
         self.bias1,
         self.weight2,
         self.bias2,
         self.weight3,
         self.bias3,
         self.weight4,
         self.bias4,
         self.weight5,
         self.bias5,
         self.weight6,
         self.bias6) = tensors

        self.groups = self.weight1.shape[0]


    def forward(self, x):
        x = x.view(-1, 28, 28)
        x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=1,
                     groups=self.groups)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, self.weight2, self.bias2, stride=1, padding=1,
                     groups=self.groups)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, self.weight3, self.bias3, stride=2, padding=1,
                     groups=self.groups)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2, padding=1,
                     groups=self.groups)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, self.weight5, self.bias5, stride=2, padding=1,
                     groups=self.groups)
        x = F.relu(x, inplace=True)
        x = F.conv2d(x, self.weight6, self.bias6, groups=self.groups)
        return x


class FunctionGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        num_mlp_units = 64
        self.mlp = nn.Sequential(
            nn.Linear(32, num_mlp_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_mlp_units, num_mlp_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_mlp_units, 9*5 + 1*5 + 32*16 + 32)
        )
        self.splits = (9, 1, 9, 1, 9, 1, 9, 1, 9, 1, 16*32, 32)

    def forward(self, x):
        tensors = self.mlp(x)

        # Split tensors into cnn tensors
        (weight1,
         bias1,
         weight2,
         bias2,
         weight3,
         bias3,
         weight4,
         bias4,
         weight5,
         bias5,
         weight6,
         bias6) = torch.split(tensors, self.splits, dim=1)

        batch_size = weight1.shape[0]

        return DynamicFunction([weight1.view(-1, 1, 3, 3),
                                bias1.flatten(),
                                weight2.view(-1, 1, 3, 3),
                                bias2.flatten(),
                                weight3.view(-1, 1, 3, 3),
                                bias3.flatten(),
                                weight4.view(-1, 1, 3, 3),
                                bias4.flatten(),
                                weight5.view(-1, 1, 3, 3),
                                bias5.flatten(),
                                # reshape happens to be necessary... TODO maybe
                                # figure out how to change to a view
                                weight6.reshape(batch_size * 32, 1, 4, 4),
                                bias6.flatten()])


class IterativeQueryConv2dModel(nn.Module):
    def __init__(self, num_iterations=2):
        super().__init__()

        # The stem doesn't grow with the batch size, so to use all the GPU
        # memory it makes sense to use more channels in the stem.
        num_stem_channels = 1
        self.stem = nn.Sequential(
            nn.Conv2d(1, num_stem_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stem_channels, num_stem_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stem_channels, num_stem_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stem_channels, num_stem_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_stem_channels, 1, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 32, 4),
        )

        self.reuse_ln = True

        use_ln = True

        if use_ln:
            self.ln_1 = nn.LayerNorm(32)
        else:
            self.ln_1 = nn.Identity()
            assert self.reuse_ln

        self.fs = nn.ModuleList([FunctionGenerator() for _ in range(num_iterations)])

        if not self.reuse_ln:
            self.lns = nn.ModuleList([nn.LayerNorm(32) for _ in range(num_iterations)])

        self.classifier = nn.Linear(32, 10)


    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        v = self.stem(x).view(-1, 32)
        v = self.ln_1(v)

        # Process the batch as a set of channels on a single image (using group convolution)
        x = x.view(1, -1, 28, 28)

        for i, f in enumerate(self.fs):
            cnn = f(v)
            if self.reuse_ln:
                ln = self.ln_1
            else:
                ln = self.lns[i]
            v = ln(cnn(x).view(-1, 32) + v)
            del cnn

        return self.classifier(v)

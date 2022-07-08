"""
For use on toy data. This code shows how to generate a batch of weights and
run it in parallel on a batch of inputs. The hypernetwork in this version is
naive, not performing any weight-sharing.
"""

import torch
from torch import nn


class DynamicFunction(nn.Module):
    def __init__(self, tensors):
        super().__init__()

        (self.weight1,
         self.bias1,
         self.weight2,
         self.bias2) = tensors

        self.batch_size = self.weight1.shape[0]

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.bmm(self.weight1, x) + self.bias1.view(self.batch_size, -1, 1)
        x = torch.bmm(self.weight2, x) + self.bias2.view(self.batch_size, -1, 1)
        return x.squeeze(-1)


class FunctionGenerator(nn.Module):
    def __init__(self, num_input_dimensions, num_state_units=4,
                 num_hidden_units=1):
        super().__init__()

        self.num_state_units = num_state_units
        self.num_hidden_units = num_hidden_units
        self.num_input_dimensions = num_input_dimensions

        self.splits = (self.num_hidden_units * self.num_input_dimensions,
                       self.num_hidden_units,
                       self.num_state_units * self.num_hidden_units,
                       self.num_state_units)

        self.mlp = nn.Sequential(
            nn.Linear(self.num_state_units, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, sum(self.splits))
        )

    def forward(self, x):
        tensors = self.mlp(x)

        # Split tensors into function tensors
        (weight1,
         bias1,
         weight2,
         bias2) = torch.split(tensors, self.splits, dim=1)

        return DynamicFunction([weight1.view(-1, self.num_hidden_units,
                                             self.num_input_dimensions),
                                bias1.flatten(),
                                weight2.view(-1, self.num_state_units,
                                             self.num_hidden_units),
                                bias2.flatten()])


class IterativeQueryVectorModel(nn.Module):
    def __init__(self, num_input_dimensions, num_classes, num_iterations=1,
                 num_state_units=4, num_hidden_units=1):
        super().__init__()

        self.num_state_units = num_state_units
        self.num_hidden_units = num_hidden_units
        self.num_input_dimensions = num_input_dimensions

        self.stem = nn.Sequential(
            nn.Linear(self.num_input_dimensions, self.num_hidden_units),
            nn.Linear(self.num_hidden_units, self.num_state_units)
        )

        self.fs = nn.ModuleList([FunctionGenerator(self.num_input_dimensions,
                                                   self.num_state_units,
                                                   self.num_hidden_units)
                                 for _ in range(num_iterations)])
        self.classifier = nn.Linear(self.num_state_units, num_classes)


    def forward(self, x):
        d = self.stem(x)

        for f in self.fs:
            dynamic_f = f(d)
            d = dynamic_f(x) + d
            del dynamic_f

        return self.classifier(d)

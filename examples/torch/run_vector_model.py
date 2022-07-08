import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from subsequent.torch.vector_model import IterativeQueryVectorModel


class GaussianVectorDataset:
    def __init__(self, num_dimensions, num_classes):
        self.num_dimensions = num_dimensions
        self._num_classes = num_classes
        self.means = torch.randn(num_classes, num_dimensions)

    def sample(self, n, shuffle=True):
        if shuffle:
            labels = torch.randint(self.num_classes(), size=(n,))
        else:
            labels = torch.arange(self.num_classes()).repeat(math.ceil(n / self.num_classes()))[:n]
        inputs = self.means[labels] + (torch.randn(n, 1) / 100)
        return inputs, labels

    def num_input_dimensions(self):
        return self.num_dimensions

    def num_classes(self):
        return self._num_classes


class LatticeDataset2D:
    """
    Gaussians centered on points of a 15x15 lattice
    """
    def __init__(self):
        r = torch.arange(-7, 8, 1, dtype=torch.float)
        self.means = torch.vstack((r.repeat_interleave(r.shape[0]),
                                   r.repeat(r.shape[0]))).T

    def sample(self, n, shuffle=True):
        if shuffle:
            labels = torch.randint(self.num_classes(), size=(n,))
        else:
            labels = torch.arange(self.num_classes()).repeat(math.ceil(n / self.num_classes()))[:n]
        inputs = self.means[labels] + (torch.randn(n, 1) / 100)
        return inputs, labels

    def num_input_dimensions(self):
        return 2

    def num_classes(self):
        return self.means.shape[0]

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, dataset, num_batches, batch_size, optimizer, criterion):
    model.train()
    for _ in tqdm(range(num_batches), leave=False):
        data, target = dataset.sample(batch_size)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, dataset, num_batches, batch_size, criterion):
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for _ in tqdm(range(num_batches), leave=False):
            data, target = dataset.sample(batch_size, shuffle=False)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / (num_batches * batch_size),
            "loss": loss / (num_batches * batch_size),
            "total_correct": total_correct}



def experiment1():
    """
    Linear classifier on 2D lattice dataset
    """
    # Training parameters
    LEARNING_RATE = 0.01
    LEARNING_RATE_GAMMA = 0.92
    MOMENTUM = 0.95
    EPOCHS = 10

    dataset = LatticeDataset2D()
    TRAIN_BATCH_SIZE = dataset.num_classes() * 5
    TEST_BATCH_SIZE = dataset.num_classes()
    TRAIN_BATCHES = 1000
    TEST_BATCHES = 1

    model = nn.Linear(2, dataset.num_classes())
#     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_RATE_GAMMA)


    for epoch in range(EPOCHS):
        train(model, dataset, TRAIN_BATCHES, TRAIN_BATCH_SIZE, optimizer, F.cross_entropy)
        lr_scheduler.step()
        results = test(model, dataset, TEST_BATCHES, TEST_BATCH_SIZE, F.cross_entropy)
        print(results)

    return model



def experiment2():
    """
    New architecture on 2D lattice dataset
    """
    # Training parameters
    LEARNING_RATE = 0.01
    LEARNING_RATE_GAMMA = 0.92
    EPOCHS = 10
    dataset = LatticeDataset2D()
    TRAIN_BATCH_SIZE = dataset.num_classes() * 5
    TEST_BATCH_SIZE = dataset.num_classes()
    TRAIN_BATCHES = 1000
    TEST_BATCHES = 1

    model = IterativeQueryConv2dModel(dataset.num_input_dimensions(), dataset.num_classes())
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_RATE_GAMMA)


    for epoch in range(EPOCHS):
        train(model, dataset, TRAIN_BATCHES, TRAIN_BATCH_SIZE, optimizer, F.cross_entropy)
        lr_scheduler.step()
        results = test(model, dataset, TEST_BATCHES, TEST_BATCH_SIZE, F.cross_entropy)
        print(results)

    return model


def experiment3():
    """
    New architecture on high dimensional gaussian dataset
    """
    # Training parameters
    LEARNING_RATE = 0.01
    LEARNING_RATE_GAMMA = 0.9
    EPOCHS = 10
    dataset = GaussianVectorDataset(1024, 4096)
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = dataset.num_classes()
    TRAIN_BATCHES = 1000
    TEST_BATCHES = 1

    model = IterativeQueryConv2dModel(dataset.num_input_dimensions(),
                                      dataset.num_classes(), num_state_units=5,
                                      num_iterations=2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_RATE_GAMMA)


    for epoch in range(EPOCHS):
        train(model, dataset, TRAIN_BATCHES, TRAIN_BATCH_SIZE, optimizer, F.cross_entropy)
        lr_scheduler.step()
        results = test(model, dataset, TEST_BATCHES, TEST_BATCH_SIZE, F.cross_entropy)
        print(results)

    return model



def experiment4():
    """
    Linear classifier with bottleneck on high dimensional gaussian dataset
    """
    # Training parameters
    LEARNING_RATE = 0.01
    LEARNING_RATE_GAMMA = 0.92
    MOMENTUM = 0.95
    EPOCHS = 10

    dataset = GaussianVectorDataset(1024, 4096)
    TRAIN_BATCH_SIZE = 1024
    TEST_BATCH_SIZE = dataset.num_classes()
    TRAIN_BATCHES = 1000
    TEST_BATCHES = 1

    model = nn.Sequential(
        nn.Linear(dataset.num_input_dimensions(), 5),
        nn.Linear(5, dataset.num_classes()),
    )
#     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LEARNING_RATE_GAMMA)


    for epoch in range(EPOCHS):
        train(model, dataset, TRAIN_BATCHES, TRAIN_BATCH_SIZE, optimizer, F.cross_entropy)
        lr_scheduler.step()
        results = test(model, dataset, TEST_BATCHES, TEST_BATCH_SIZE, F.cross_entropy)
        print(results)

    return model



if __name__ == "__main__":
    experiment4()

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from subsequent.torch.conv2d_model import IterativeQueryConv2dModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(loader, leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, loader, criterion):
    model.eval()
    loss = 0
    total_correct = 0
    with torch.no_grad():
        for data, target in tqdm(loader, leave=False):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def experiment1():


    MEAN = 0.13062755
    STDEV = 0.30810780
    folder = os.path.expanduser("~/transient/datasets")
    train_dataset = datasets.MNIST(
        folder, train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((MEAN,), (STDEV,))]))
    test_dataset = datasets.MNIST(
        folder, train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((MEAN,), (STDEV,))]))

    # Training parameters
    LEARNING_RATE = 0.0005 # 0.001
    LEARNING_RATE_GAMMA = 0.3
    EPOCHS = 30
    TRAIN_BATCH_SIZE = 100
    TEST_BATCH_SIZE = 1000
    WEIGHT_DECAY = 0

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE,
                                              shuffle=False)

    model = IterativeQueryConv2dModel()
    model = model.to(device)
    print(model)
    print("# of parameters: ", count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=LEARNING_RATE_GAMMA)

    for epoch in range(EPOCHS):
        train(model=model, loader=train_loader, optimizer=optimizer, criterion=F.cross_entropy)
        lr_scheduler.step()
        results = test(model=model, loader=test_loader, criterion=F.cross_entropy)
        print(results)

    return model


if __name__ == "__main__":
    experiment1()

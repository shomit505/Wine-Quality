import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from typing import Type, Tuple

class Train():
    def __init__(self, inputs: pd.DataFrame, targets: pd.DataFrame, net_class: Type[object], loss_fn, train_split: float, lr: float, k_fold: int, batch_size: int, epochs: int, seed: int) -> None:
        self.train_split = train_split
        self.test_split = 1 - train_split
        self.k_fold = k_fold
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.targets = targets
        self.dataset_size = inputs.shape[0]
        self.net_class = net_class
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        torch.manual_seed(seed)
        randomised_inputs = torch.tensor(self.inputs.to_numpy(), dtype=torch.float32)[torch.randperm(self.dataset_size)]
        randomised_targets = torch.tensor(self.targets.to_numpy(), dtype=torch.float32)[torch.randperm(self.dataset_size)]
        train_cut = round(self.train_split * self.dataset_size)

        self.train_inputs = randomised_inputs[0:train_cut]
        self.train_targets = randomised_targets[0:train_cut]
        self.test_inputs = randomised_inputs[train_cut:]
        self.test_targets = randomised_targets[train_cut:]

    def optimise(self, inputs, targets, epochs):
        trainset = TensorDataset(inputs, targets)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=False)
        optimiser = optim.Adam(self.net_class.parameters(), lr=0.01)

        for _ in range(epochs):
            for X_batch, y_batch in trainloader:
                outputs = self.net_class.forward(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        
        return 'Training Done!'
    
    def evaluate(self, inputs, targets):
        return self.loss_fn(torch.round(self.net_class.forward(inputs)), targets)
    
    def crossvalidate(self) -> float:
        running_loss = 0
        for i in range(self.k_fold):
            train_indices = [j for j in range(len(self.train_inputs)) if j % self.k_fold != i]
            test_indices = [j for j in range(len(self.train_inputs)) if j % self.k_fold == i]
            if self.loss_fn is None:
                weights = self.net_class.erm(self.train_inputs[train_indices], self.train_targets[train_indices])
                test_loss = self.net_class.loss(self.train_inputs[test_indices], self.train_targets[test_indices], weights)
            else:
                _ = self.optimise(self.train_inputs[train_indices], self.train_targets[train_indices], self.epochs)
                test_loss = self.evaluate(self.train_inputs[test_indices], self.train_targets[test_indices])
            running_loss += test_loss

        return running_loss / self.k_fold
    
    def finaltrain(self) -> Tuple[float, torch.Tensor]:
        if self.loss_fn is None:
            weights = self.net_class.erm(self.train_inputs, self.train_targets)
            test_loss = self.net_class.loss(self.test_inputs, self.test_targets, weights)
        else:
            _ = self.optimise(self.train_inputs, self.train_targets, self.epochs)
            test_loss = self.evaluate(self.test_inputs, self.test_targets)

        return test_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from typing import Type, Tuple

class LLS():
    def erm(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        print(inputs.shape)
        return torch.linalg.inv(inputs.T @ inputs) @ inputs.T @ targets
    
    def loss(self, inputs, targets, weights) -> torch.Tensor:
        return (targets - inputs @ weights).T @ (targets - inputs @ weights) / len(inputs)
    

class RIDGELLS():
    def __init__(self, regulariser: float) -> None:
        self.regulariser = regulariser

    def erm(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(inputs.T @ inputs + self.regulariser * np.identity(inputs.shape[1])) @ inputs.T @ targets
    
    def loss(self, inputs, targets, weights) -> torch.Tensor:
        return ((targets - inputs @ weights).T @ (targets - inputs @ weights) + self.regulariser * weights.T @ weights) / len(inputs)
    
class KernelRIDGE():
    def __init__(self, regulariser: float, gamma: float) -> None:
        self.regulariser = regulariser
        self.gamma = gamma

    def erm(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = inputs[:, None, :] - inputs[None, :, :]  
        K = np.exp(- self.gamma * np.sum(diff**2, axis=-1))
        return inputs.T @ torch.linalg.inv(K + np.identity(len(K))) @ targets
    
    def loss(self, inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return ((targets - inputs @ weights).T @ (targets - inputs @ weights) + self.regulariser * weights.T @ weights) / len(inputs)
    
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        prediction = self.net(x)
        return prediction
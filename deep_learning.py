import argparse
from ucimlrepo import fetch_ucirepo
import torch.nn as nn
from functions.function_approximators import NeuralNet
from train.train import Train


wine_quality = fetch_ucirepo(id=186) 
X = wine_quality.data.features 
y = wine_quality.data.targets

net_object = NeuralNet()
loss_fn = nn.MSELoss()

if __name__ ="__main__":
    argparse.ArgumentParser(description="Training a neural net to predict wine quality")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train/ Test split")
    parser.add_argument("--k_fold", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--batch_size", type=int, default=32. help="batch size")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs for training")
    args = parser.parse_args()

    Train = Train(X, y, net_object, loss_fn, args.train_split, args.learning_rate, args.k_fold, args.batch_size, args.epochs)
    Train.finaltrain()
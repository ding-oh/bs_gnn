import os, sys
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
from tqdm import tqdm
from utils      import os_utils
from data_utils import readDataset, preprocessBS
import random
import joblib
from generateBindingsiteGraph import *

# def main():

# define path location
proj_p = '/home/dongwoo/project/geo/BS_prediction'
data_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'

validCnt, invalidCnt, numTrainSet = 0, 0, 1 # using set0 for test
data_list = []

for i in range(numTrainSet):
    print(f'[INFO ] set {i}')
    for idx, seqInfo in enumerate(open(f'{proj_p}/valid_data.set/set{i}_seq.info', 'r').readlines()[1:]):
        try:
            result = convertBS2Graph(seqInfo=seqInfo,
                            file_p=f'{data_p}/set_{i}/{seqInfo.split(",")[0]}',
                            pdb=seqInfo.split(',')[0],
                            chain=seqInfo.split(',')[1]
                            )
            edge_index, node_attr, edge_attr, edge_weight = result
            y = torch.tensor([(float(seqInfo.split(',')[5]))])
            dtmp = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, y=y)
            data_list.append(dtmp)
            print("data_preparation :", idx, '.', seqInfo.split(',')[0], seqInfo.split(',')[1], "...ok")
        except Exception as E:
            print(E)

random.seed(12345)
random.shuffle(data_list)

train_set, val_set, test_set = data_list[:-300], data_list[-300:-200], data_list[-300:]
print(f"Number of training set: {len(train_set)}")
print(f"Number of Validaation set: {len(val_set)}")
print(f"Number of test set: {len(test_set)}")

train_loader = DataLoader(train_set, batch_size=100, shuffle=True, drop_last=False)
test_loader = DataLoader(test_set, batch_size=100, shuffle=True, drop_last=False)
val_loader = DataLoader(val_set, batch_size=100, shuffle=False, drop_last=False)

data = data_list[0]
print("data.keys : ", data.keys)

data.num_node_features

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)
model = myGCN(in_channel=int(data.num_node_features), hidden_layer_size=70)
model.to(device)
print(model)
count_parameters(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()

def test(loader):
    model.eval()
    error = 0.0
    out_all = []
    true = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.to(device))
        tmp = (out - data.y) ** 2
        error += tmp.sum().item()  # Check against ground-truth labels.

        out_all.extend([x.item() for x in out])
        true.extend([x.item() for x in data.y])

    return error / len(loader.dataset), out_all, true  # Derive ratio of correct predictions.

def train():
    model.train()
    for idx, batch in enumerate(train_loader):
        out = model(batch.to(device))
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        if idx % 10 == 0:
            print(f"Loss: {loss:.4f}")

train_acc_list = []
val_acc_list = []
test_acc_list = []

for epoch in range(1, 180):
    print("=" * 100)
    print("Epoch: ", epoch)

    train()

    train_acc, out_tmp, true_tmp = test(train_loader)
    train_acc_list.append(train_acc)

    test_acc, out_all, true_all = test(test_loader)
    test_acc_list.append(test_acc)

    val_acc, val_pred, val_true = test(val_loader)
    val_acc_list.append(val_acc)

    print("-" * 100)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# if __name__ == '__main__':
#     main()
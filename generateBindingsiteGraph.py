import os, sys
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from prettytable import PrettyTable
from tqdm import tqdm

from utils      import os_utils
from data_utils import readDataset, preprocessBS

def convertBS2Graph(seqInfo, file_p, pdb, chain):
    """
    convert protein to graph based on sequence, dssp, ASA
    """

    seq, bs_binary = seqInfo.split(',')[3], seqInfo.split(',')[-1]

    # get other information
    onehot_seq         = preprocessBS.getAAonehot(seq)
    dssp_seq, sasa_seq = preprocessBS.getStructuralInfo(file_p, pdb, chain)
    
    assert onehot_seq.shape[0] == len(dssp_seq) == len(sasa_seq)

    node_attr = []
    for idx, AA in enumerate(seq):
        hydropathy = preprocessBS.getAminoAcidPathy(AA) # hydrophobic, hydrophilic, amphipathic
        
        # 21-dim + 3-dim + 9-dim + 1-dim = 34-dim
        attr = list(onehot_seq[idx]) + hydropathy + list(dssp_seq[idx]); attr.append(sasa_seq[idx])
        node_attr.append(attr)

    # Contact Map
    edge_attr, edge_index, edge_weight = preprocessBS.GetContactmapInfo(file_p, pdb, chain)

    # Preparation
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_attr = torch.tensor(node_attr, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    return node_attr, edge_attr, edge_index, edge_weight


import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class myGCN(torch.nn.Module):
    def __init__(self, in_channel=37, hidden_layer_size=70):
        super().__init__()
        self.conv1 = GCNConv(in_channel,
                             hidden_layer_size)  # 가장 기본적인 graph convolution model, https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        self.conv2 = GCNConv(hidden_layer_size, hidden_layer_size)
        self.conv3 = GCNConv(hidden_layer_size, hidden_layer_size)
        self.lin1 = Linear(hidden_layer_size, int(hidden_layer_size / 2))
        self.lin2 = Linear(int(hidden_layer_size / 2), 1)

    def forward(self, data):
        x, edge_index, batch, edge_weight = data.x, data.edge_index, data.batch, data.edge_weight

        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.gelu(x)

        x = self.conv3(x, edge_index)
        x = F.gelu(x)

        # READOUT
        x = global_mean_pool(x, batch)  # 전체의 node feature의 평균 값을 취한다. # [batch_size, hidden_channels]
        x = self.lin1(x)  # 70 dim -> 35-dim
        x = F.elu(x)
        x = self.lin2(x)  # 35 dim -> 1-dim

        return x

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    # define path location
    proj_p = '/home/dongwoo/project/geo/BS_prediction'
    data_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'
    
    validCnt, invalidCnt, numTrainSet = 0, 0, 9
    for i in range(numTrainSet): 
        print(f'[INFO ] set {i}')
        for idx, seqInfo in enumerate(open(f'{proj_p}/valid_data.set/set{i}_seq.info', 'r').readlines()[1:]):
            try:
                convertBS2Graph(seqInfo=seqInfo,
                                file_p=f'{data_p}/set_{i}/{seqInfo.split(",")[0]}',
                                pdb=seqInfo.split(',')[0],
                                chain=seqInfo.split(',')[1]
                )

            except Exception as E:
                print(E)

# if __name__ == '__main__':
#     main()

import os, sys
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
import torch

import pandas as pd

#import torch
#import torch.nn.fuctional as F

def getAAonehot(seq):
    AA_dict = {
        'W': 0, 'R': 1, 'H': 2, 'K': 3, 'D': 4, 'E': 5,
        'S': 6, 'T': 7, 'N': 8, 'Q': 9, 'C': 10,
        'U': 11, 'G': 12, 'P': 13, 'A': 14, 'V': 15,
        'I': 16, 'L': 17, 'M': 18, 'F': 19, 'Y': 20, 
    }

    int_seq = [ AA_dict[i] for i in seq ]
    numAAs  = len(AA_dict.keys())
    onehot_seq = np.array([ np.eye(numAAs)[i] for i in int_seq ])
    
    assert len(int_seq) == onehot_seq.shape[0]
    return onehot_seq

def getAminoAcidPathy(AA):
    if AA in ['G', 'P', 'F', 'A', 'I', 'L', 'V']: # hydrophobic
        return [1, 0, 0]
    elif AA in ['S', 'T', 'D', 'E', 'C', 'N', 'Q', 'R', 'H', 'U']: # hydrophilic
        return [0, 1, 0]
    elif AA in ['K', 'W', 'Y', 'M']: # amphipathic
        return [0, 0, 1]
    else:
        print('Wrong amino acid type')
        sys.exit(1)


def getStructuralInfo(file_p, pdb, chain):
    dssp_dict = {
        'T': 0, 'E': 1, 'G': 2, 'B': 3, 'S': 4, 'I': 5,
        'X': 6, 'H': 7, 'C': 8
    }
    numDsspType = len(dssp_dict.keys())

    dssp_seq, sasa_seq = [], []
    for i in open(f'{file_p}/{pdb}{chain}_str.dat', 'r').readlines():
        secStruct  = np.eye(numDsspType)[ dssp_dict[i.split()[2]] ]
        solventasa = float(i.split()[3])

        dssp_seq.append(secStruct)
        sasa_seq.append(solventasa)
    return dssp_seq, sasa_seq


proj_p = '/home/dongwoo/project/geo/BS_prediction'
data_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'
for i in range(9):
    for idx, seqInfo in enumerate(open(f'{proj_p}/valid_data.set/set{i}_seq.info', 'r').readlines()[1:]):
            seqInfo=seqInfo
            file_p=f'{data_p}/set_{i}/{seqInfo.split(",")[0]}'
            pdb=seqInfo.split(',')[0]
            chain=seqInfo.split(',')[1]

def GetContactmapInfo(file_p, pdb, chain):

    map_size = max([int(i.split()[1]) for i in open(f'{file_p}/{pdb}{chain}_cnt.dat', 'r')])
    map = np.zeros((map_size, map_size))

    for j in open(f'{file_p}/{pdb}{chain}_cnt.dat', 'r').readlines():
        alpha = j.split()[0]
        beta = j.split()[1]
        value_attr = j.split()[2]

        #edge_attr
        edge_attr = map
        edge_attr[int(alpha) - 1, int(beta) - 1] = value_attr
        edge_attr[int(beta) - 1, int(alpha) - 1] = value_attr

        #edge_index
        edge_index = np.where(map > 0, 1, map)

        #edge_weight
        edge_weight = np.reciprocal(edge_attr)
        edge_weight = np.nan_to_num(edge_weight, copy=True, posinf=0)

    return edge_attr, edge_index, edge_weight

# def GetContactmapInfo(file_p, pdb, chain):
#
#     map_size = max([int(i.split()[1]) for i in open(f'{file_p}/{pdb}{chain}_cnt.dat', 'r')])
#     map = np.zeros((map_size, map_size))
#
#     for j in open(f'{file_p}/{pdb}{chain}_cnt.dat', 'r').readlines():
#         alpha = j.split()[0]
#         beta = j.split()[1]
#         value_attr = j.split()[2]
#
#         #edge_attr
#         edge_attr = map
#         edge_attr[int(alpha) - 1, int(beta) - 1] = value_attr
#         edge_attr[int(beta) - 1, int(alpha) - 1] = value_attr
#
#         #edge_index
#         edge_index = np.where(map > 0, 1, map)
#
#         #edge_weight
#         edge_weight = np.reciprocal(edge_attr)
#         edge_weight = np.nan_to_num(edge_weight, copy=True, posinf=0)

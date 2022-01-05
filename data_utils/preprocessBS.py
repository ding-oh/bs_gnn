import os, sys
import numpy as np
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
    # Soy-based adhesives for wood-bonding?
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


def getEdgeCAcontact(file_p, pdb, chain, seq_len):

    edge_idx, edge_weight = [], np.zeros( (seq_len, seq_len) )
    edge_weight = []
    for i in open(f'{file_p}/{pdb}{chain}_cnt.dat', 'r').readlines():
        ca_i, ca_j, dist = int(i.split()[0])-1, int(i.split()[1])-1,  1 / ( 1 + np.log(float(i.split()[2])) )

        # generate undirected-graph edge index
        edge_idx.append( [ca_i, ca_j] ); edge_idx.append( [ca_j, ca_i] )

        # generate edge weight based on the distance function
        #ca_contact[ca_i,ca_j], ca_contact[ca_j,ca_i] = dist, dist
        edge_weight.extend( [dist, dist] )

    return edge_idx, edge_weight





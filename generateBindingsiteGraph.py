import os, sys
import numpy as np
import pandas as pd

# import torch
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
    
    return node_attr, bs_binary
    







def main():
    # define path location
    proj_p = '/home/byun/PROject/BS_prediction'
    data_p = '/home/juyong/projects_at_data1/scPDB_m/clean_set'
    
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




if __name__ == '__main__':
    main()

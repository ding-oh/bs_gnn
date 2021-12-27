import os, sys
import numpy as np
import pandas as pd


from tqdm import tqdm

from utils      import os_utils
from data_utils import readDataset, preprocessBS 



def convertBS2Graph(data_f):
    for line in open(data_f).readlines()[1:]:
        seq = line.split(',')[3]
        onehot_seq = preprocessBS.onehot_AA(seq)
        print(onehot_seq)


def main():
    # define path location
    proj_p = '/home/dongwoo/project/geo/BS_prediction'
    data_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'
    # distance_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'

    validCnt, invalidCnt = 0, 0
    for i in range(9): # iterate for loop as much as No. of set
        print(f'[INFO ] set {i}')
        if os_utils.chkFileDirExists(path_f=f'{proj_p}/set{i}_seq.info', _file=True):
            pass
        else:
            # write binding site information
            f = open(f'{proj_p}/set{i}_seq.info', 'w')
            f.write('pdb,chain,seq_len,seq,secondary_str,bs_binary\n')
            set_p = f'{data_p}/set_{i}'
    
            for pdb in tqdm(os.listdir(set_p)):  # pdb id from set
                for j in os.listdir(f'{set_p}/{pdb}'):  # find fasta file
                    if j.endswith('.fa'):
                        try:
                            seqInfo = readDataset.getSequenceInfo(f'{set_p}/{pdb}/{j}')
                            f.write(f'{pdb},{j[4]},{str(seqInfo[0])},{str(seqInfo[1])},{str(seqInfo[2])},{str(seqInfo[3])}\n')
                            validCnt += 1
                        except Exception as e:
                            f.write(f"{pdb} didn't have sequence file {e}")
                            invalidCnt += 1
            f.close()


if __name__ == '__main__':
    main()

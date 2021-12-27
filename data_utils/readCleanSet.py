import os, sys
import numpy as np
import pandas as pd
import preprocessBS
from tqdm import tqdm


def getSecondaryInfo(str_f):
    str_dssp = ''
    for i in open(str_f).readlines():
        str_dssp += str(i.split()[2].strip())
    return str_dssp

def getContactInfo(cnt_f):
    max_residue = max([int(i.split()[1]) for i in open(f'{cnt_f}', 'r')])
    return max_residue

def getSequenceInfo(fa_f):
    seq = open(fa_f, 'r').readlines()[1][:-1] # omit '\n'
    str_dssp  = getSecondaryInfo(f'{fa_f.split(".fa")[0]}_str.dat')
    bs_binary = open(f'{fa_f.split(".fa")[0]}_bs.dat', 'r').readline()[:-1] # omit '\n'
    max_residue = getContactInfo(f'{fa_f.split(".fa")[0]}_cnt.dat')

    assert len(seq) == len(str_dssp) == len(bs_binary) == max_residue, f'sequence lengths are different {len(seq)}/ {len(str_dssp)}/ {len(bs_binary)}/'
    return len(seq), seq, str_dssp, bs_binary, max_residue

def main():
    # define path location
    proj_p = '/home/dongwoo/project/geo/BS_prediction'
    data_p = '/home/dongwoo/project/geo/scPDB_m/clean_set'
    
    validCnt, invalidCnt = 0, 0
    for i in range(9): # No. of set
        print(f'[INFO ] set {i}')
        f = open(f'{proj_p}/set{i}_seq.info', 'w')
        f.write('pdb,chain,seq_len,seq,secondary_str,bs_binary')
        set_p = f'{data_p}/set_{i}'

        # for pdb in tqdm(os.listdir(set_p)): # pdb id from set
        #     for j in os.listdir(f'{set_p}/{pdb}'): # find fasta file
        #         if j.endswith('.fa'):
        #             try:
        #                 seqInfo = getSequenceInfo(f'{set_p}/{pdb}/{j}')
        #                 f.write(f'{pdb},{j[4]},{str(seqInfo[0])},{str(seqInfo[1])},{str(seqInfo[2])},{str(seqInfo[3])}\n')
        #                 validCnt += 1
        #             except Exception as e:
        #                 f.write(f"{pdb} didn't have sequence file {e}")
        #                 invalidCnt += 1
        f.close()

    print(f'[INFO ] valid Cnt: {validCnt} | invalid Cnt: {invalidCnt}')


if __name__ == '__main__':
    main()

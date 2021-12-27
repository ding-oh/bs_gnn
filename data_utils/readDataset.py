import os, sys
import numpy as np
import pandas as pd

def getSecondaryInfo(str_f):
    str_dssp = ''
    for i in open(str_f).readlines():
        str_dssp += str(i.split()[2].strip())
    return str_dssp

def getSequenceInfo(fa_f):
    seq = open(fa_f, 'r').readlines()[1][:-1] # omit '\n'
    str_dssp  = getSecondaryInfo(f'{fa_f.split(".fa")[0]}_str.dat')
    bs_binary = open(f'{fa_f.split(".fa")[0]}_bs.dat', 'r').readline()[:-1] # omit '\n'
    assert len(seq) == len(str_dssp) == len(bs_binary), f'sequence lengths are different {len(seq)}/ {len(str_dssp)}/ {len(bs_binary)}'
    return len(seq), seq, str_dssp, bs_binary

def getContactInfo(fa_f):
    cnt = open(f'{fa_f.split(".cnt")[0]}_cnt.dat', 'r').readlines()[:-1] # omit '\n'
    return cnt

import os
from data_utils import readDataset, preprocessBS


def countmap(cnt_f):
    result_list = []
    for (root, directories, files) in os.walk(cnt_f):
        for file in files:
            if '_bs.dat' in file:
               with open(f'{root}/{file}','r', encoding='cp949') as f:
                   try:
                       lines = f.read().splitlines()
                       result_list.append(lines)
                   except:
                       pass
countmap()


# for edge_idx in range(fa_f):
import os, sys
import copy



def chkFileDirExists(path_f=None, path_d=None, _file=None, _dir=None):
    if _file == True and os.path.exists(f'{path_f}'):
        return True
    elif _dir == True and os.path.exists(f'{path_d}'):
        return True
    else:
        return False

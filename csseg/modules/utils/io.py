'''
Function:
    Implementation of some utils for io-related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import pickle


'''touchdir'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
            return True
        except:
            return False
    return False


'''saveckpts'''
def saveckpts(ckpts, savepath):
    torch.save(ckpts, savepath)
    return True


'''loadckpts'''
def loadckpts(ckptspath, map_to_cpu=True):
    if map_to_cpu:
        ckpts = torch.load(ckptspath, map_location=torch.device('cpu'))
    else:
        ckpts = torch.load(ckptspath)
    return ckpts


'''saveaspickle'''
def saveaspickle(data, savepath):
    with open(savepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True


'''loadpicklefile'''
def loadpicklefile(filepath):
    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data


'''symlink'''
def symlink(src_path, dst_path):
    if os.path.exists(dst_path):
        os.unlink(dst_path)
    os.symlink(src_path, dst_path)
    return True
'''
Function:
    Implementation of some utils for io-related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import pickle
import torch.utils.model_zoo as model_zoo


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
    if os.path.islink(ckptspath):
        ckptspath = os.readlink(ckptspath)
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
    if os.path.islink(dst_path):
        os.unlink(dst_path)
    os.symlink(src_path, dst_path)
    return True


'''loadpretrainedweights'''
def loadpretrainedweights(structure_type, pretrained_model_path='', pretrained_weights_table={}, map_to_cpu=True, possible_model_keys=['model', 'state_dict']):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        checkpoint = torch.load(pretrained_model_path, map_location='cpu') if map_to_cpu else torch.load(pretrained_model_path)
    else:
        checkpoint = model_zoo.load_url(pretrained_weights_table[structure_type], map_location='cpu') if map_to_cpu else model_zoo.load_url(pretrained_weights_table[structure_type])
    state_dict = checkpoint
    for key in possible_model_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    return state_dict
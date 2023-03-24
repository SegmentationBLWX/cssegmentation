'''
Function:
    Implementation of Convertor for activation functions
Author:
    Zhenchao Jin
'''
import torch


'''actname2torchactname'''
def actname2torchactname(act_name):
    # supported act names
    supported_actnames = {
        'ReLU': 'relu',
        'Tanh': 'tanh',
        'LeakyReLU': 'leaky_relu',
        'SELU': 'selu',
    }
    # return
    return supported_actnames[act_name]
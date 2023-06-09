'''
Function:
    Implementation of BaseDataset
Author:
    Zhenchao Jin
'''
import os
import copy
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from .pipelines import SegmentationEvaluator
from .pipelines import (
    Compose, Resize, CenterCrop, Pad, Lambda, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip,
    ToTensor, Normalize, RandomCrop, RandomResizedCrop, ColorJitter
)


'''Subset'''
class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transforms=None, seg_target_transforms=None):
        # set attributes
        self.dataset = dataset
        self.indices = indices
        self.transforms = transforms
        self.seg_target_transforms = seg_target_transforms
    '''getitem'''
    def __getitem__(self, index):
        data_meta = self.dataset[self.indices[index]]
        data_meta = self.transforms(data_meta) if self.transforms is not None else data_meta
        if 'seg_target' in data_meta and data_meta['seg_target'] is not None:
            data_meta['seg_target'] = self.seg_target_transforms(data_meta['seg_target']) if self.seg_target_transforms is not None else data_meta['seg_target']
        return data_meta
    '''len'''
    def __len__(self):
        return len(self.indices)


'''_BaseDataset'''
class _BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, dataset_cfg):
        # assert
        assert mode in ['TRAIN', 'TEST']
        # set attributes
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.transforms = self.constructtransforms(dataset_cfg.get('transforms'))
        self.imageids, self.image_dir, self.ann_dir = [], '', ''
    '''getitem'''
    def __getitem__(self, index):
        # prepare
        imageid = self.imageids[index]
        imagepath = os.path.join(self.image_dir, f'{imageid}.jpg')
        annpath = os.path.join(self.ann_dir, f'{imageid}.png')
        # read image and seg_target
        image, seg_target = np.array(Image.open(imagepath).convert('RGB'))[::-1], None
        if self.mode == 'TRAIN': assert os.path.exists(annpath)
        if os.path.exists(annpath):
            seg_target = np.array(Image.open(annpath))
        # perform transforms
        data_meta = {
            'image': image, 'seg_target': seg_target, 'imageid': imageid,
            'width': image.size[0], 'height': image.size[1],
        }
        data_meta = self.transforms(data_meta) if self.transforms is not None else data_meta
        # return
        return data_meta
    '''len'''
    def __len__(self):
        return len(self.imageids)
    '''constructtransforms'''
    @staticmethod
    def constructtransforms(transform_settings):
        if transform_settings is None: return transform_settings
        # supported transforms
        supported_transforms = {
            'Resize': Resize, 'CenterCrop': CenterCrop, 'Pad': Pad, 'Lambda': Lambda, 
            'RandomRotation': RandomRotation, 'RandomHorizontalFlip': RandomHorizontalFlip, 
            'RandomVerticalFlip': RandomVerticalFlip, 'ToTensor': ToTensor, 'Normalize': Normalize, 
            'RandomCrop': RandomCrop, 'RandomResizedCrop': RandomResizedCrop, 'ColorJitter': ColorJitter,
        }
        # construct transforms
        transforms = []
        for transform_setting in transform_settings:
            name, params = transform_setting
            transforms.append(supported_transforms[name](**params))
        # return
        return Compose(transforms)


'''BaseDataset'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, task_name, task_id, dataset_cfg):
        # assert
        assert mode in ['TRAIN', 'TEST']
        # set attributes
        self.mode = mode
        self.task_id = task_id
        self.task_name = task_name
        self.dataset_cfg = dataset_cfg
        dataset_cfg_g = copy.deepcopy(dataset_cfg)
        dataset_cfg_g.pop('transforms')
        self.data_generator = self.builddatagenerator(mode, dataset_cfg_g)
        self.num_classes = self.data_generator.num_classes
        self.transforms = self.data_generator.constructtransforms(dataset_cfg['transforms'])
        # prepare for training
        self.prepare(dataset_cfg, self.transforms, self.data_generator)
    '''getitem'''
    def __getitem__(self, index):
        return self.data_generator[index]
    '''builddatagenerator'''
    def builddatagenerator(self, mode, dataset_cfg):
        raise NotImplementedError('not to be implemented')
    '''prepare'''
    def prepare(self, dataset_cfg, transforms, data_generator):
        # parse
        labels, history_labels = self.gettasklabels(task_name=self.task_name, tasks=self.tasks, task_id=self.task_id)
        if self.mode == 'TEST': 
            labels = history_labels + labels
            history_labels = None
        overlap, masking_value = dataset_cfg['overlap'], dataset_cfg['masking_value']
        # filter images
        history_labels = history_labels if history_labels is not None else []
        self.stripzero(labels)
        self.stripzero(history_labels)
        assert not any(l in history_labels for l in labels)
        self.labels = [0] + labels
        self.history_labels = [0] + history_labels
        self.all_labels = [0] + history_labels + labels
        selected_indices = self.filterimages(data_generator, labels, history_labels, overlap)
        # remap the labels
        self.labels_to_trainlabels_map = {label: self.all_labels.index(label) for label in self.all_labels}
        self.labels_to_trainlabels_map[255] = 255
        seg_target_transforms = torchvision.transforms.Lambda(
            lambda t: t.apply_(lambda x: self.labels_to_trainlabels_map[x] if x in self.labels + [255] else masking_value)
        )
        # obtain subset
        self.data_generator = Subset(data_generator, selected_indices, transforms, seg_target_transforms)
    '''gettasklabels'''
    @staticmethod
    def gettasklabels(task_name, tasks, task_id):
        labels = list(tasks[task_name][task_id])
        history_labels = [label for s in range(task_id) for label in tasks[task_name][s]]
        return labels, history_labels
    '''getnumclassespertask'''
    @staticmethod
    def getnumclassespertask(task_name, tasks, task_id):
        num_classes_per_task = [len(tasks[task_name][s]) for s in range(task_id + 1)]
        return num_classes_per_task
    '''filterimages'''
    @staticmethod
    def filterimages(dataset, labels, history_labels=None, overlap=True):
        selected_indices = []
        if 0 in labels: labels.remove(0)
        if history_labels is None: history_labels = []
        all_labels = labels + history_labels + [0, 255]
        if overlap:
            filter_func = lambda c: any(x in labels for x in cls)
        else:
            filter_func = lambda c: any(x in labels for x in cls) and all(x in all_labels for x in c)
        if torch.distributed.get_rank() == 0:
            pbar = tqdm(dataset)
            pbar.set_description('Filtering Images')
        else:
            pbar = dataset
        for idx, data_meta in enumerate(pbar):
            cls = np.unique(np.array(data_meta['seg_target']))
            if filter_func(cls):
                selected_indices.append(idx)
        return selected_indices
    '''stripzero'''
    @staticmethod
    def stripzero(labels):
        while 0 in labels:
            labels.remove(0)
    '''evaluate'''
    @staticmethod
    def evaluate(self, seg_gts, seg_preds, device=None):
        seg_evaluator = SegmentationEvaluator(num_classes=self.data_generator.num_classes)
        seg_evaluator.update(seg_gts=seg_gts, seg_preds=seg_preds)
        seg_evaluator.synchronize(device=device)
        return seg_evaluator.evaluate()
    '''len'''
    def __len__(self):
        return len(self.data_generator)
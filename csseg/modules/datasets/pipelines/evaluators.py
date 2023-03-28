'''
Function:
    Implementation of Evaluators
Author:
    Zhenchao Jin
'''
import torch
import numpy as np
import torch.distributed as dist


'''SegmentationEvaluator'''
class SegmentationEvaluator():
    def __init__(self, num_classes, eps=1e-6):
        self.eps = eps
        self.num_classes = num_classes
        self.reset()
    '''reset'''
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    '''synchronize'''
    def synchronize(self, device=None):
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        total_samples = torch.tensor(self.total_samples).to(device)
        dist.reduce(confusion_matrix, dst=0)
        dist.reduce(total_samples, dst=0)
        self.confusion_matrix = confusion_matrix.cpu().numpy()
        self.total_samples = total_samples.cpu().numpy()
    '''update'''
    def update(self, seg_targets, seg_preds):
        for st, sp in zip(seg_targets, seg_preds):
            self.confusion_matrix += self.fasthist(st.flatten(), sp.flatten())
        self.total_samples += len(seg_targets)
    '''fasthist'''
    def fasthist(self, seg_target, seg_pred):
        mask = (seg_target >= 0) & (seg_target < self.num_classes)
        hist = np.bincount(
            self.num_classes * seg_target[mask].astype(int) + seg_pred[mask], minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        return hist
    '''evaluate'''
    def evaluate(self):
        # obtain variables
        eps = self.eps
        hist = self.confusion_matrix
        # evaluate
        all_accuracy = np.diag(hist).sum() / hist.sum()
        mean_accuracy = np.mean((np.diag(hist) / (hist.sum(axis=1) + eps))[hist.sum(axis=1) != 0])
        iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + eps)
        mean_iou = np.mean(iou[hist.sum(axis=1) != 0])
        class_iou = dict(zip(range(self.num_classes), [iou[i] if m else 'INVALID' for i, m in enumerate(hist.sum(axis=1) != 0)]))
        class_accuracy = dict(zip(range(self.num_classes), [(np.diag(hist) / (hist.sum(axis=1) + eps))[i] if m else 'INVALID' for i, m in enumerate(hist.sum(axis=1) != 0)]))
        # summarize
        results = {
            'all_accuracy': all_accuracy, 'mean_accuracy': mean_accuracy, 'mean_iou': mean_iou,
            'class_iou': class_iou, 'class_accuracy': class_accuracy
        }
        # return
        return results
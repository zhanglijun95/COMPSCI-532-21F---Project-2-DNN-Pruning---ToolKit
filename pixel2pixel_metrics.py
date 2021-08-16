import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DataMetrics(object):
    def __init__(self, task):
        super(DataMetrics, self).__init__()
        if task not in ['segment_semantic', 'depth_zbuffer']:
            print('Wrong task name! Please choose segment_semantic or depth_zbuffer.')
            exit()
        self.task = task
        
    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        
    def reset_records(self):
        self.batch_size = []
        if self.task is 'segment_semantic':
            self.records = {'mIoUs': [], 'pixelAccs': [],  'errs': [], 'conf_mat': np.zeros((self.num_seg_cls, self.num_seg_cls)), 'labels': np.arange(self.num_seg_cls)}
        elif self.task is 'depth_zbuffer':
            self.records = {'abs_errs': [], 'rel_errs': []}
        
    def resize_pred(self, pred, gt):
        return F.interpolate(pred, size=gt.shape[-2:])
    
    def __seg_error(self, pred, gt):
        output = self.resize_pred(pred, gt)
        
        gt = gt.view(-1)
        labels = gt < self.num_seg_cls
        gt = gt[labels].int()

        logits = output.permute(0, 2, 3, 1).contiguous().view(-1, self.num_seg_cls)
        logits = logits[labels]
        err = self.cross_entropy(logits, gt.long())

        prediction = torch.argmax(output, dim=1)
        prediction = prediction.unsqueeze(1)
        prediction = prediction.view(-1)
        prediction = prediction[labels].int()
        pixelAcc = (gt == prediction).float().mean()
        return prediction.cpu().detach().numpy(), gt.cpu().detach().numpy(), pixelAcc.cpu().detach().numpy(), err.cpu().detach().numpy()
    
    def __seg_records(self, pred, gt):
        pred, gt, pixelAcc, err = self.__seg_error(pred, gt)
        new_mat = confusion_matrix(gt, pred, self.records['labels'])
        self.records['conf_mat'] += new_mat
        self.records['pixelAccs'].append(pixelAcc)
        self.records['errs'].append(err)
        
    def __depth_error(self, pred, gt, mask):
        output = self.resize_pred(pred, gt)
        
        if mask is not None:
            binary_mask = (gt != 255) * (mask.int() == 1)
        else:
            binary_mask = (torch.sum(gt, dim=1) > 3 * 1e-5).unsqueeze(1).cuda()
        
        depth_output_true = output.masked_select(binary_mask)
        depth_gt_true = gt.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        return abs_err.cpu().detach().numpy(), rel_err.cpu().detach().numpy()
    
    def __depth_records(self, pred, gt, mask):
        abs_err, rel_err, sq_rel_err, ratio, rms, rms_log = self.__depth_error(pred, gt, mask)
        self.records['abs_errs'].append(abs_err)
        self.records['rel_errs'].append(rel_err)
    
    # Call for each batch
    def __call__(self, pred, gt, mask=None):
        self.batch_size.append(len(gt))
        
        if self.task is 'segment_semantic':
            self.__seg_records(pred, gt)
        elif self.task is 'depth_zbuffer':
            self.__depth_records(pred, gt, mask)
        return
    
    # Helper function
    def round_dict(self, d):
        res = {key : round(d[key], 4) for key in d}
        return res
    
    def depth_records_modify(self):
        self.records['abs_errs'] = np.stack(self.records['abs_errs'], axis=0)
        self.records['rel_errs'] = np.stack(self.records['rel_errs'], axis=0)
        return
    
class CityScapesMetrics(DataMetrics):
    def __init__(self, task):
        super(CityScapesMetrics, self).__init__(task)
        if self.task is 'segment_semantic':
            self.num_seg_cls = 19
        
        self.define_loss()
        self.define_refer()
        self.reset_records()
        
    def define_loss(self):
        super(CityScapesMetrics, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        
    def define_refer(self):
        if self.task is 'segment_semantic':
            self.refer = {'mIoU': 0.427, 'Pixel Acc': 0.681}
        elif self.task is 'depth_zbuffer':
            self.refer = {'abs_err': 0.026, 'rel_err': 0.39}
        
     # Call after evaluate all data in the set
    def val_metrics(self):
        if self.task is 'segment_semantic':
            val_metrics = self.__seg_metrics()
        elif self.task is 'depth_zbuffer':
            val_metrics = self.__depth_metrics()
        self.reset_records()
        return self.round_dict(val_metrics)
    
    # Private functions
    def __seg_metrics(self):
        val_metrics = {}
        jaccard_perclass = []
        for i in range(self.num_seg_cls):
            if not self.records['conf_mat'][i, i] == 0:
                jaccard_perclass.append(self.records['conf_mat'][i, i] / (np.sum(self.records['conf_mat'][i, :]) + 
                                                       np.sum(self.records['conf_mat'][:, i]) -
                                                       self.records['conf_mat'][i, i]))

        val_metrics['mIoU'] = np.sum(jaccard_perclass) / len(jaccard_perclass)
        val_metrics['Pixel Acc'] = (np.array(self.records['pixelAccs']) * np.array(self.batch_size)).sum() / sum(self.batch_size)
        
#         val_metrics['cmp'] = (((val_metrics['mIoU'] - self.refer['mIoU'] ) / self.refer['mIoU']) +
#                        ((val_metrics['Pixel Acc'] - self.refer['Pixel Acc'] ) / self.refer['Pixel Acc'])) /2 
        return val_metrics
    
    
    def __depth_metrics(self):
        val_metrics = {}
        self.depth_records_modify()
        val_metrics['abs_err'] = (self.records['abs_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        val_metrics['rel_err'] = (self.records['rel_errs'] * np.array(self.batch_size)).sum() / sum(self.batch_size)
        
#         val_metrics['cmp'] = (((self.refer['abs_err'] - val_metrics['abs_err']) / self.refer['abs_err']) + 
#                        ((self.refer['rel_err'] - val_metrics['rel_err']) / self.refer['rel_err'])) /2
        return val_metrics
    

      
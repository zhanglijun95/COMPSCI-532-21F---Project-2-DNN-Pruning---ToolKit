import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataCriterions(nn.Module):
    def __init__(self, task):
        super(DataCriterions, self).__init__()
        if task not in ['segment_semantic', 'depth_zbuffer']:
            print('Wrong task name! Please choose segment_semantic or depth_zbuffer.')
            exit()
        self.task = task
        
    def define_loss(self):
        self.cosine_similiarity = nn.CosineSimilarity()
        self.l1_loss = nn.L1Loss()
        self.l1_loss_sum = nn.L1Loss(reduction='sum')
        
    def seg_loss(self, pred, gt):
        prediction = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_seg_cls)
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape).permute(0, 2, 3, 1).contiguous().view(-1)
        loss = self.cross_entropy(prediction, gt.long())
        return loss
    
    def depth_loss(self, pred, gt, mask=None):
        new_shape = pred.shape[-2:]
        gt = F.interpolate(gt.float(), size=new_shape) 
        if mask is not None:
            gt_mask = F.interpolate(mask.float(), size=new_shape)
            binary_mask = (gt != 255) * (gt_mask.int() == 1)
        else:
            binary_mask = (torch.sum(gt, dim=1) > 3 * 1e-5).unsqueeze(1).cuda()
        prediction = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        loss = self.l1_loss(prediction, gt)
        return loss

    def forward(self, pred, gt, mask=None):
        if self.task is 'segment_semantic':
            return self.seg_loss(pred, gt)
        elif self.task is 'depth_zbuffer':
            return self.depth_loss(pred, gt, mask)
        
class CityScapesCriterions(DataCriterions):
    def __init__(self, task):
        super(CityScapesCriterions, self).__init__(task)
        if self.task is 'segment_semantic':
            self.num_seg_cls = 19
        self.define_loss()
        
    def define_loss(self):
        super(CityScapesCriterions, self).define_loss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        
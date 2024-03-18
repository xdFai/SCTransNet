import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, preds, gt_masks):
        if isinstance(preds, list):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() - intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        elif isinstance(preds, tuple):
            loss_total = 0
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() - intersection.sum() + smooth)
                loss = 1 - loss.mean()
                a.append(loss)
            # loss_total = a[0] * 0.1 + a[1] * 0.2 + a[2] * 0.3 + a[3] * 0.4 + a[4] * 0.5 + a[5]
            loss_total = a[0] * 0.1 + a[1] * 0.2 + a[2] * 0.3 + a[3] * 0.4 + a[4]
            return loss_total
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() - intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss


class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()

    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())

        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)

        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt) + self.softiou(preds[1].sigmoid(), edge_gt)

        return loss_img + loss_edge

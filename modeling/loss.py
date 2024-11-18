import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts

    weights = np.ones(7)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()

class FocalLoss_Revised(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1e-6):
        """
        Initialize the Combined Dice and Focal Loss function.

        Args:
            alpha (float): Balancing factor for focal loss. Default is 0.25.
            gamma (float): Focusing parameter for focal loss. Default is 2.0.
            smooth (float): Smoothing factor to avoid division by zero in Dice loss.
        """
        super(FocalLoss_Revised, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, output, target):
        """
        Compute the focal loss.

        Args:
            logits (torch.Tensor): Raw predictions from the model (B, C, H, W).
            targets (torch.Tensor): Ground truth labels (B, C, H, W).

        Returns:
            torch.Tensor: Focal loss.
        """
         # Apply sigmoid to logits
        probs = torch.sigmoid(output)
        
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))

        # Focal Loss
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * target * torch.log(probs + self.smooth)
        focal_loss += -(1 - self.alpha) * probs ** self.gamma * (1 - target) * torch.log(1 - probs + self.smooth)
        focal_loss = focal_loss.mean()
        return focal_loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss

class CombinedDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3.0, smooth=1e-6):
        """
        Initialize the Combined Dice and Focal Loss function.

        Args:
            alpha (float): Balancing factor for focal loss. Default is 0.25.
            gamma (float): Focusing parameter for focal loss. Default is 2.0.
            smooth (float): Smoothing factor to avoid division by zero in Dice loss.
        """
        super(CombinedDiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, output, target):
        """
        Compute the combined loss.

        Args:
            logits (torch.Tensor): Raw predictions from the model (B, C, H, W).
            targets (torch.Tensor): Ground truth labels (B, C, H, W).

        Returns:
            torch.Tensor: Combined Dice and Focal loss.
        """
         # Apply sigmoid to logits
        probs = torch.sigmoid(output)
        
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))

        # Dice Loss
        # intersection = (output_flat * target_flat).sum(dim=1)
        intersection = (output_flat * target_flat).sum()
        dice_loss = 1 - ((2.0 * intersection + self.smooth) /
                         (output_flat.sum() + target_flat.sum() + self.smooth)).mean()

        # Focal Loss
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * target * torch.log(probs + self.smooth)
        focal_loss += -(1 - self.alpha) * probs ** self.gamma * (1 - target) * torch.log(1 - probs + self.smooth)
        focal_loss = focal_loss.mean()

        # Combined Loss
        combined_loss = dice_loss + focal_loss
        return combined_loss

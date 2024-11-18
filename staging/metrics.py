import numpy as np
import torch
from config import EvalConfig

def batch_pix_accuracy(predict, target, labeled):
    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()
    # return pixel_correct.numpy(), pixel_labeled.numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    print(num_class)
    return area_inter.cpu().numpy(), area_union.cpu().numpy()

def eval_metrics(output, target, num_class, metric):
    _, predict = torch.max(output.data, 1)
    predict = predict + 1
    target = target + 1

    labeled = (target > 0) * (target <= num_class)
    if metric == 'batch_pix_accuracy':
        correct, num_labeled = batch_pix_accuracy(predict, target, labeled)
        numerator, denominator = correct, num_labeled
    elif metric == 'batch_intersection_union':
        inter, union = batch_intersection_union(predict, target, num_class, labeled)
        numerator, denominator = inter, union
    if metric == 'weighted_f1':
        numerator = batch_f1_score(
            predict, target, num_class, labeled, class_weights=EvalConfig.F1_SCORE_CLASS_WEIGHT
        )
        denominator = 1   # For one batch
    return numerator, denominator

def get_epoch_acc(batch_acc_numerator, batch_acc_denominator, metric):
    if metric == 'batch_pix_accuracy':
        total_correct = batch_acc_numerator
        total_label = batch_acc_denominator
        pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        acc = pixAcc
    elif metric == 'batch_intersection_union':
        total_inter = batch_acc_numerator
        total_union = batch_acc_denominator
        IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        mIoU = IoU.mean()
        acc = mIoU
    if metric == 'weighted_f1':
        acc = batch_acc_numerator/(1 + batch_acc_denominator)
        
    return acc


def batch_f1_score(predict, target, num_class, labeled, class_weights=None):
    """
    Computes the weighted F1 score for image segmentation on a batch of predictions and targets.
    
    Args:
        predict (torch.Tensor): Predicted labels (batch, height, width).
        target (torch.Tensor): Ground truth labels (batch, height, width).
        num_class (int): Number of classes (including background).
        labeled (torch.Tensor): Binary mask indicating labeled pixels.
        class_weights (list or torch.Tensor): Weights for each class (default: equal weights).
    
    Returns:
        torch.Tensor: Weighted F1 scores for each class (shape: [num_class]).
    """
    predict = predict * labeled.long()
    target = target * labeled.long()
    
    f1_scores = []
    for class_id in range(num_class):  # Class IDs are from 0 to num_class-1
        pred_mask = (predict == class_id)
        target_mask = (target == class_id)
        
        tp = (pred_mask & target_mask).sum().float()  # True Positive
        fp = (pred_mask & ~target_mask).sum().float()  # False Positive
        fn = (~pred_mask & target_mask).sum().float()  # False Negative

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        f1_scores.append(f1)
    
    f1_scores = torch.tensor(f1_scores)
    
    # Apply class weights
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        weighted_f1 = (f1_scores * class_weights).sum()
        weighted_f1 = torch.sigmoid(weighted_f1)  # Normalize the result to range [0, 1]
    else:
        weighted_f1 = f1_scores  # Equal weight if no weights provided
    
    return weighted_f1.item()

def eval_f1(output, target, num_class, class_weights=None):
    """
    Wrapper to evaluate weighted F1 score over a batch.
    
    Args:
        output (torch.Tensor): Model logits (batch, num_classes, height, width).
        target (torch.Tensor): Ground truth labels (batch, height, width).
        num_class (int): Number of classes (including background).
        class_weights (list or torch.Tensor): Weights for each class (default: equal weights).
    
    Returns:
        float: Weighted average F1 score across classes.
    """
    _, predict = torch.max(output.data, 1)  # Get predicted class IDs
    target = target
    
    labeled = (target >= 0) & (target < num_class)  # Exclude invalid labels
    f1_scores = batch_f1_score(predict, target, num_class, labeled, class_weights)
    
    # Compute weighted average F1 score
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        weighted_avg_f1 = (f1_scores * class_weights).sum() / class_weights.sum()
    else:
        weighted_avg_f1 = f1_scores.mean()  # Unweighted average
    
    return weighted_avg_f1.item()

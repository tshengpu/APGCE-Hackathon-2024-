import numpy as np
import torch

def batch_pix_accuracy(predict, target, labeled):
    # pixel_labeled = labeled.sum()
    # pixel_correct = ((predict == target) * labeled).sum()
    # assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    # return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()
    # # return pixel_correct.numpy(), pixel_labeled.numpy()

    pixel_labeled = labeled.sum()
    pixel_correct = ((predict == target) * labeled).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()

def batch_intersection_union(predict, target, num_class, labeled):
    predict = predict * labeled.long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
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
    return acc
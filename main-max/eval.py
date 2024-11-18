from loss import *
from metrics import eval_metrics, get_epoch_acc

def eval(model, class_count, criterion, eval_metric, device, dataloader, print_metrics = False):
    model.eval()
    batch_loss = 0
    batch_acc_numerator = 0
    batch_acc_denominator = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        print('########')
        print(inputs.shape)
        print('########')
        
        mask_pred = model(inputs)
        loss = criterion(mask_pred, labels)

        # batch_loss += loss
        batch_loss += loss.item()

        
        batch_acc_numerator_tmp, batch_acc_denominator_tmp = eval_metrics(mask_pred, labels, class_count, eval_metric)
        batch_acc_numerator += batch_acc_numerator_tmp
        batch_acc_denominator += batch_acc_denominator_tmp

    epoch_loss = batch_loss / len(dataloader)
    epoch_acc = get_epoch_acc(batch_acc_numerator, batch_acc_denominator, eval_metric)

    if print_metrics:
        print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

    return epoch_loss, epoch_acc
import numpy as np
import torch

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def mean_iou(label_trues, label_preds, n_class, return_all=False):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.00001)
    
    mean_iu = np.nanmean(iu[1:])
    weight = hist.sum(axis=1)[1:]/sum(hist.sum(axis=1)[1:])
    mean_iou_weight = sum(iu[1:] * weight)
    if return_all:
        return iu, weight, mean_iou_weight

    return mean_iou_weight


def mean_iou_bg(label_trues, label_preds, n_class, return_all=False):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.00001)
    
    
    mean_iu = np.nanmean(iu)
    weight = hist.sum(axis=1)/sum(hist.sum(axis=1))
    mean_iou_weight = sum(iu * weight)
    if return_all:
        return iu, weight, mean_iou_weight

    return mean_iou_weight


def get_hist(label_trues, label_preds, n_class):
    """
    """
    hist = np.zeros((n_class, n_class))
    hist += _fast_hist(label_trues.flatten(), label_preds.flatten(), n_class)
    return hist

# from https://github.com/hszhao/semseg/blob/master/util/util.py
def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union

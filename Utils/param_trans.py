import torch
import numpy as np
import torch.nn.functional as F


def trans_target(target, num_classes):
    assert len(target.shape) == 4 or len(target.shape) == 5
    if len(target.shape) == 4:
        target = torch.nn.functional.one_hot(target, num_classes).permute(0, 4, 1, 2, 3)
    return target


def trans_data(data):
    assert len(data.shape) == 3 or len(data.shape) == 4
    if len(data.shape) == 3:
        data = data[None, ...]
    return data


def trans_label(label):
    assert isinstance(label, torch.Tensor) or isinstance(label, np.ndarray)
    if isinstance(label, np.ndarray):
        label = torch.from_numpy(label).long()
    return label


def trans_type(data):
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    return data


def trans_predict(predict):
    assert len(predict.shape) == 4 or len(predict.shape) == 5
    if len(predict.shape) == 5:
        predict = F.softmax(predict, dim=1).max(dim=1)[1]
    predict = trans_type(predict)
    return predict

import torch
import numpy as np
import torch.nn.functional as F


def trans_target(target, num_classes):
    assert len(target.shape) == 4 or len(target.shape) == 5
    if len(target.shape) == 4:
        # target = torch.nn.functional.one_hot(target, num_classes).type(target.dtype).permute(0, 4, 1, 2, 3)
        target = torch.nn.functional.one_hot(target, num_classes).permute(0, 4, 1, 2, 3)

    return target


# def trans_target_loss(target, num_classes):
#     assert len(target.shape) == 4 or len(target.shape) == 5
#     if len(target.shape) == 4:
#         # target = torch.nn.functional.one_hot(target, num_classes).type(target.dtype).permute(0, 4, 1, 2, 3)
#         target = target.unsqueeze(dim=1)
#
#     return target


def trans_data(data):
    assert len(data.shape) == 3 or len(data.shape) == 4
    if len(data.shape) == 3:
        data = data[None, ...]
    return data


def trans_type(data):
    assert isinstance(data, torch.Tensor) or isinstance(data, np.ndarray)
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if 0 == np.count_nonzero(data):
        data[0, 0, 0, 0] = 1

    # unique, count = np.unique(data, return_counts=True)
    # data_count = dict(zip(unique, count))
    # print(data_count)

    return data


def trans_predict(predict):
    assert len(predict.shape) == 4 or len(predict.shape) == 5
    if len(predict.shape) == 5:
        predict = F.softmax(predict, dim=1).max(dim=1)[1]
    predict = trans_type(predict)
    # avoid predict all zero
    return predict

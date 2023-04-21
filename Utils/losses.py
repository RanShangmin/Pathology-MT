import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import monai
from Utils import ramps
from Utils.param_trans import trans_target


class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                   ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, d, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()
        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, d, h, w)

        return self.criterion(pred, target)


class consistency_weight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, final_w, iters_per_epoch, rampup_starts=0, rampup_ends=100, ramp_type='sigmoid_rampup'):
        self.final_w = final_w
        self.iters_per_epoch = iters_per_epoch
        self.rampup_starts = rampup_starts * iters_per_epoch
        self.rampup_ends = rampup_ends * iters_per_epoch
        self.rampup_length = (self.rampup_ends - self.rampup_starts)
        self.rampup_func = getattr(ramps, ramp_type)
        self.current_rampup = 0

    def __call__(self, epoch, curr_iter):
        cur_total_iter = self.iters_per_epoch * epoch + curr_iter
        if cur_total_iter < self.rampup_starts:
            return 0
        self.current_rampup = self.rampup_func(cur_total_iter - self.rampup_starts, self.rampup_length)
        return self.final_w * self.current_rampup


def dice_loss(predict, target, num_classes=2):
    # unique, count = np.unique(target.cpu().numpy(), return_counts=True)
    # data_count = dict(zip(unique, count))
    # print("target: ",data_count)

    target = trans_target(target, num_classes=num_classes)
    criterion = monai.losses.DiceCELoss(sigmoid=True, lambda_ce=1.0, lambda_dice=1.0)

    # unique, count = np.unique(predict.detach().max(1)[1].cpu().numpy(), return_counts=True)
    # data_count = dict(zip(unique, count))
    # print("predict: ",data_count)

    # unique, count = np.unique(target.cpu().numpy(), return_counts=True)
    # data_count = dict(zip(unique, count))
    # print("target: ",data_count)

    return criterion(predict, target)


def semi_dice_loss(inputs, targets,
                   conf_mask=True, threshold=None,
                   threshold_neg=None, num_classes=2, temperature_value=1):
    # target => logit, input => logit
    # assert len(targets.shape) == 4 or len(targets.shape) == 5
    # targets = trans_target(targets, num_classes=num_classes)
    pass_rate = {}
    if conf_mask:
        # for negative
        targets_prob = F.softmax(targets / temperature_value, dim=1)

        # for positive
        targets_real_prob = F.softmax(targets, dim=1)

        weight = targets_real_prob.max(1)[0]
        total_number = len(targets_prob.flatten(0))
        boundary = ["< 0.1", "0.1~0.2", "0.2~0.3",
                    "0.3~0.4", "0.4~0.5", "0.5~0.6",
                    "0.6~0.7", "0.7~0.8", "0.8~0.9",
                    "> 0.9"]

        rate = [torch.sum((torch.logical_and((i - 1) / 10 < targets_real_prob, targets_real_prob < i / 10)) == True)
                / total_number for i in range(1, 11)]

        max_rate = [torch.sum((torch.logical_and((i - 1) / 10 < weight, weight < i / 10)) == True)
                    / weight.numel() for i in range(1, 11)]

        pass_rate["entire_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, rate)]
        pass_rate["max_prob_boundary"] = [[label, val] for (label, val) in zip(boundary, max_rate)]

        mask = (weight >= threshold)

        mask_neg = (targets_prob < threshold_neg)

        # temp negative label * mask_neg, which mask down the positive labels.
        # neg_label = torch.ones(targets.shape, dtype=targets.dtype, device=targets.device) * mask_neg
        neg_label = trans_target(torch.argmax(targets_prob, dim=1), num_classes=num_classes)
        # if neg_label.shape[-1] != num_classes:
        #     neg_label = torch.cat((neg_label, torch.zeros([neg_label.shape[0], neg_label.shape[1],
        #                                                    neg_label.shape[2], neg_label.shape[3],
        #                                                    num_classes - neg_label.shape[-1]]).cuda()),
        #                           dim=4)
        # neg_label = neg_label.permute(0, 4, 1, 2, 3)
        neg_label = 1 - neg_label

        if not torch.any(mask):
            neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))
            # zero = torch.tensor(0., dtype=torch.float, device=negative_loss_mat.device)
            return inputs.sum() * .0, pass_rate, negative_loss_mat[mask_neg].mean()
        else:
            targets = trans_target(torch.argmax(F.softmax(targets, dim=1), dim=1), num_classes=num_classes)
            loss = monai.losses.DiceCELoss(sigmoid=True)
            positive_loss_mat = loss(inputs, targets).mean()
            # positive_loss_mat = F.cross_entropy(inputs, torch.argmax(targets, dim=1), reduction="none")

            positive_loss_mat = positive_loss_mat * weight

            neg_prediction_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-7, max=1.)
            negative_loss_mat = -(neg_label * torch.log(neg_prediction_prob))

            return positive_loss_mat[mask].mean(), pass_rate, negative_loss_mat[mask_neg].mean()
    else:
        raise NotImplementedError

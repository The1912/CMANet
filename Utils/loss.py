import torch
import numpy as np
from torch import nn


med_frq_sunrgbd = [0.382900, 0.452448, 0.637584, 0.377464, 0.585595,
                   0.479574, 0.781544, 0.982534, 1.017466, 0.624581,
                   2.589096, 0.980794, 0.920340, 0.667984, 1.172291,
                   0.862240, 0.921714, 2.154782, 1.187832, 1.178115,
                   1.848545, 1.428922, 2.849658, 0.771605, 1.656668,
                   4.483506, 2.209922, 1.120280, 2.790182, 0.706519,
                   3.994768, 2.220004, 0.972934, 1.481525, 5.342475,
                   0.750738, 4.040773]

med_frq_NYU2 = [0.71142, 0.77194, 0.81391, 0.94143, 0.95208, 1.07680, 1.14535, 1.14909, 1.10488,
                1.17025, 1.15129, 1.38882, 1.28413, 1.63873, 1.44600, 1.64278, 1.72670, 1.87923,
                1.57468, 2.17806, 1.81955, 1.33333, 2.46872, 2.50839, 2.41232, 3.29888, 3.56601,
                3.05714, 3.02347, 4.33856, 3.85831, 4.45739, 4.07171, 4.66439, 4.24873, 5.12041,
                5.27822, 1.09183, 1.18313, 0.87580]


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq_NYU2):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

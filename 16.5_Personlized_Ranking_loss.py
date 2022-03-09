import torch
import torch.nn as nn

class HingeLossbRec(nn.Module):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        loss = nn.MultiMarginLoss()
        return loss(positive, negative)

class BPRLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BPRLoss, self).__init__(**kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - torch.sum(torch.log(self.sigmoid(distances)))
        return loss
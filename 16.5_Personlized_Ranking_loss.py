import torch
import torch.nn as nn

class HingeLossbRec(nn.Module):
    '''
    https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html
    '''
    def __init__(self, **kwargs):
        super(HingeLossbRec, self).__init__(**kwargs)

    def forward(self, positive, negative, margin=1):
        loss = nn.HingeEmbeddingLoss(margin = margin)
        return loss(positive, negative)

class BPRLoss(nn.Module):
    def __init__(self, **kwargs):
        super(BPRLoss, self).__init__(**kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - torch.sum(torch.log(self.sigmoid(distances)))
        return loss
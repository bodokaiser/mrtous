import numpy as np
import torch
import torch.nn
import torch.autograd

class ElasticLoss(torch.nn.Module):

    def __init__(self, beta):
        super().__init__()

        self.beta = beta

    def forward(self, input1, input2):
        diff = input1 - input2

        loss = self.beta*diff.norm(2)
        loss += (1-self.beta)*diff.norm(1)

        return loss

class ThresholdedElasticLoss(torch.nn.Module):

    def __init__(self, beta):
        super().__init__()

        self.beta = beta

    def forward(self, input1, input2):
        vec1 = input1.clone().contiguous().view(-1)
        vec2 = input2.clone().contiguous().view(-1)

        thr1 = vec1.mean()-2*vec1.var()
        thr2 = vec2.mean()-2*vec2.var()

        value = torch.min(thr1, thr2)
        value.clamp(0, 1)
        index = torch.max(vec1.lt(value.data[0]), vec2.lt(value.data[0]))
        index.clamp(0, 1).byte()

        vec1[index] = 0
        vec2[index] = 0

        return ElasticLoss(self.beta)(input1, input2)

def elastic_loss(input1, input2, beta=.5):
    return ElasticLoss(beta)(input1, input2)

def thresholded_elastic_loss(input1, input2, beta=.5):
    return ThresholdedElasticLoss(beta)(input1, input2)
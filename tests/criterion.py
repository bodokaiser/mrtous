import torch
import unittest

from mrtous import criterion, dataset
from torch.autograd import Variable

class TestElasticLoss(unittest.TestCase):

    def setUp(self):
        self.x = Variable(torch.randn(3, 3))
        self.y = Variable(torch.randn(3, 3))

    def test_forward(self):
        l1_loss = self.x.dist(self.y, 1)
        l2_loss = self.x.dist(self.y, 2)
        loss = criterion.elastic_loss(self.x, self.y)

        self.assertEqual(loss, .5*l1_loss+.5*l2_loss)

class TestThresholdedElasticLoss(unittest.TestCase):

    def setUp(self):
        self.x = Variable(torch.FloatTensor([
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        ])).unsqueeze(1).unsqueeze(1)
        self.y = Variable(torch.FloatTensor([
            [[1, 1, 1], [0, 0, 0], [0, 0, -1]],
            [[1, 1, 1], [0, 0, 0], [0, 0, -1]],
        ])).unsqueeze(1).unsqueeze(1)
        self.z = Variable(torch.FloatTensor([
            0, 1, 2,
            0, 0, 0,
            0, 0, 0,
        ]))

    def test_forward(self):
        l1_loss = self.z.norm(1)
        l2_loss = self.z.norm(2)
        loss = criterion.thresholded_elastic_loss(self.x, self.y)

        self.assertEqual(loss, l1_loss+l2_loss)
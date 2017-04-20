import numpy as np

from unittest import TestCase

from mrtous.dataset import MINC2, MNIBITE

class TestMINC2(TestCase):

    def setUp(self):
        self.mr_minc = MINC2('data/01_mr.mnc')
        self.us_minc = MINC2('data/01_us.mnc')

    def test_getlen(self):
        self.assertEqual(len(self.mr_minc), 378)
        self.assertEqual(len(self.us_minc), 378)

    def test_getitem(self):
        self.assertTupleEqual(self.mr_minc[0].shape, (466, 394))
        self.assertTupleEqual(self.us_minc[0].shape, (466, 394))
        self.assertTupleEqual(self.mr_minc[:, 0].shape, (378, 394))
        self.assertTupleEqual(self.mr_minc[:, 0].shape, (378, 394))
        self.assertTupleEqual(self.mr_minc[:, :, 0].shape, (378, 466))
        self.assertTupleEqual(self.mr_minc[:, :, 0].shape, (378, 466))


class TestMNIBITE(TestCase):

    def setUp(self):
        self.mnibite = MNIBITE('data/01_mr.mnc', 'data/01_us.mnc')

    def test_init(self):
        self.assertIsInstance(self.mnibite.mr, MINC2)
        self.assertIsInstance(self.mnibite.us, MINC2)

    def test_getlen(self):
        self.assertEqual(len(self.mnibite), 378)
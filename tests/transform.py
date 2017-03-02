import numpy as np
import unittest

from mrtous import dataset, transform

class TestRegionCrop(unittest.TestCase):

    US = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], np.float32)

    MR = np.ones([5, 5], np.float32)

    def setUp(self):
        self.mr = dataset.MINC2('mnibite/01_mr.mnc')
        self.us = dataset.MINC2('mnibite/01_us.mnc')
        self.transform = transform.RegionCrop()

    def test_transform(self):
        with self.subTest():
            mr, us = self.transform(self.MR, self.US)
            self.assertFalse(np.any(mr))
            self.assertFalse(np.any(us))
        with self.subTest():
            mr, us = self.transform(self.mr[0], self.us[0])
            np.testing.assert_array_equal(mr, self.mr[0])
            np.testing.assert_array_equal(us, self.us[0])
        with self.subTest():
            mr, us = self.transform(self.mr[120], self.us[120])
            self.assertTupleEqual(mr.shape, us.shape)
            self.assertLess(mr.shape[0], self.mr[120].shape[0])
            self.assertLess(mr.shape[1], self.mr[120].shape[1])
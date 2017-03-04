import unittest
import numpy as np

from mrtous import dataset

class TestMINC2(unittest.TestCase):

    LENGTH = [394, 466, 378]
    RANGES = [[-32768, 32767], [0, 255]]

    def setUp(self):
        self.minc = [
            dataset.MINC2('mnibite/01_mr.mnc'),
            dataset.MINC2('mnibite/01_us.mnc'),
        ]

    def test_init(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                self.assertEqual(list(minc.vrange), self.RANGES[index])
                self.assertEqual(minc.xlength, self.LENGTH[0])
                self.assertEqual(minc.ylength, self.LENGTH[1])
                self.assertEqual(minc.zlength, self.LENGTH[2])

    def test_iter(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                for _ in minc:
                    pass

    def test_getlen(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                self.assertEqual(len(minc), np.sum(self.LENGTH))

    def test_getitem(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                self.assertLessEqual(minc[0].max(), 1.0)
                self.assertGreaterEqual(minc[0].min(), 0.0)

                np.testing.assert_array_equal(minc[0], minc.volume[0])
                np.testing.assert_array_equal(minc[self.LENGTH[2]-1],
                    minc.volume[self.LENGTH[2]-1])

                np.testing.assert_array_equal(minc[self.LENGTH[2]],
                    np.flipud(minc.volume[:, 0]))
                np.testing.assert_array_equal(minc[np.sum(self.LENGTH[1:3])-1],
                    np.flipud(minc.volume[:, self.LENGTH[1]-1]))

                np.testing.assert_array_equal(minc[np.sum(self.LENGTH[1:3])],
                    np.flipud(minc.volume[:, :, 0]))
                np.testing.assert_array_equal(minc[np.sum(self.LENGTH)-1],
                    np.flipud(minc.volume[:, :, self.LENGTH[0]-1]))

                with self.assertRaises(IndexError):
                    minc[np.sum(self.LENGTH)]

class TestMNIBITENative(unittest.TestCase):

    def setUp(self):
        self.mnibite = dataset.MNIBITENative('mnibite', 1)

    def test_init(self):
        self.assertIsInstance(self.mnibite.mr, dataset.MINC2)
        self.assertIsInstance(self.mnibite.us, dataset.MINC2)

    def test_getlen(self):
        self.assertEqual(len(self.mnibite), len(self.mnibite.mr))
        self.assertEqual(len(self.mnibite), len(self.mnibite.us))

    def test_getitem(self):
        def transform(mr, us):
            np.testing.assert_array_equal(mr, self.mnibite.mr[0])
            np.testing.assert_array_equal(us, self.mnibite.us[1])
            return 1, 2
        np.testing.assert_array_equal(self.mnibite[0][0], self.mnibite.mr[0])
        np.testing.assert_array_equal(self.mnibite[0][1], self.mnibite.us[0])
        with self.assertRaises(IndexError):
            self.mnibite[len(self.mnibite)]
        self.mnibite.transform = transform
        self.assertTupleEqual(self.mnibite[0], (1, 2))
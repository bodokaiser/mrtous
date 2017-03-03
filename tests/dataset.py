import unittest

from numpy import testing
from mrtous import dataset

class TestMINC2(unittest.TestCase):

    VALID_LENGTH = [394, 466, 378]
    VALID_RANGES = [[-32768, 32767], [0, 255]]

    def setUp(self):
        self.minc = [
            dataset.MINC2('mnibite/01_mr.mnc'),
            dataset.MINC2('mnibite/01_us.mnc'),
        ]

    def test_init(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                self.assertEqual(list(minc.vrange), self.VALID_RANGES[index])
                self.assertEqual(minc.xlength, self.VALID_LENGTH[0])
                self.assertEqual(minc.ylength, self.VALID_LENGTH[1])
                self.assertEqual(minc.zlength, self.VALID_LENGTH[2])

    def test_iter(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                for _ in minc:
                    pass

    def test_getlen(self):
        self.assertEqual(len(self.minc[0]), len(self.minc[0].volume))
        self.assertEqual(len(self.minc[1]), len(self.minc[1].volume))

        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                self.assertEqual(len(minc), self.VALID_LENGTH[2])

    def test_getitem(self):
        for index, minc in enumerate(self.minc):
            with self.subTest(index=index):
                self.assertLessEqual(minc[0].max(), +1.0)
                self.assertGreaterEqual(minc[0].min(), -1.0)

                with self.assertRaises(IndexError):
                    minc[self.VALID_RANGES[2]]

class TestMNIBITE(unittest.TestCase):

    def setUp(self):
        self.mnibite = dataset.MNIBITE('mnibite', 1)

    def test_init(self):
        self.assertIsInstance(self.mnibite.mr, dataset.MINC2)
        self.assertIsInstance(self.mnibite.us, dataset.MINC2)

    def test_getlen(self):
        self.assertEqual(len(self.mnibite), len(self.mnibite.mr))
        self.assertEqual(len(self.mnibite), len(self.mnibite.us))

    def test_getitem(self):
        def transform(mr, us):
            testing.assert_array_equal(mr, self.mnibite.mr[0])
            testing.assert_array_equal(us, self.mnibite.us[1])
            return 1, 2
        testing.assert_array_equal(self.mnibite[0][0], self.mnibite.mr[0])
        testing.assert_array_equal(self.mnibite[0][1], self.mnibite.us[0])
        with self.assertRaises(IndexError):
            self.mnibite[len(self.mnibite)]
        self.mnibite.transform = transform
        self.assertTupleEqual(self.mnibite[0], (1, 2))
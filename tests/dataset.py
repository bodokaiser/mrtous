from unittest import TestCase

from mrtous.dataset import MINC2, MNIBITE

class TestMINC2(TestCase):

    def setUp(self):
        self.mr_minc = MINC2('data/mr/01.mnc')
        self.us_minc = MINC2('data/us/01.mnc')

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
        self.mnibite = MNIBITE('data')
        self.mr_minc1 = MINC2('data/mr/01.mnc')
        self.mr_minc2 = MINC2('data/mr/02.mnc')

    def test_init(self):
        self.assertEqual(len(self.mnibite.mr), 14)
        self.assertEqual(len(self.mnibite.us), 14)

    def test_getlen(self):
        self.assertEqual(len(self.mnibite), 14*378)

    def test_getitem(self):
        with self.assertRaises(IndexError):
            self.mnibite[378*14]
        self.assertEqual(self.mnibite[0][0].sum(), self.mr_minc1[0].sum())
        self.assertEqual(self.mnibite[377][0].sum(), self.mr_minc1[377].sum())
        self.assertEqual(self.mnibite[378][0].sum(), self.mr_minc2[0].sum())
        self.assertEqual(self.mnibite[755][0].sum(), self.mr_minc2[377].sum())
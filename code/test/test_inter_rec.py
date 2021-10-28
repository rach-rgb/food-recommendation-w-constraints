import unittest
from sys import path
from surprise import SVD

# setting path
path.append('../code')

import code.inter_rec as inter_rec


class DummyAlgo(SVD):
    def __init__(self):
        SVD.__init__(self)

        self.i_attr = None
        self.const = None
        self.n_err = None

    def set_data(self, attr, const, n_err):
        self.i_attr = attr
        self.const = const
        self.n_err = n_err


class TestPostRec(unittest.TestCase):

    def setUp(self):
        self.dummy = DummyAlgo()
        self.rec = inter_rec.InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.dummy)
        self.rec.get_data()
        self.rec.train()
        self.rec.test()

    # load required data
    def test_get_data(self):
        # check algo() object attr
        const = self.dummy.const.loc[self.rec.const.u == 0].iloc[0]
        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], "")

        # check algo() object const
        attr = self.dummy.i_attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 200)

        # check algo() n_err
        self.assertEqual(self.rec.n_err, self.dummy.n_err)

    # test sort_prediction()
    def test_sort_predictions(self):
        self.rec.sort_predictions()
        top_n = self.rec.top_N

        self.assertEqual(3, len(top_n[3]))
        self.assertEqual([0, 1, 2], sorted(top_n[2]))

    # test sort_prediction slice ratings
    def test_sort_predictions2(self):
        self.rec.result_N = 1
        self.rec.sort_predictions()
        top_n = self.rec.top_N

        self.assertEqual(1, len(top_n[3]))

    # test top_n_const_1
    # caution: dummy algorithm isn't doesn't constraints
    def test_top_n_const_1(self):
        self.assertEqual([1, 3], sorted(self.rec.top_n_const_1(0, 1)))
        self.assertEqual(0, len(self.rec.top_n_const_1(0, 0)))

    # test top_n_const_2
    # caution: dummy algorithm doesn't constraints
    def test_top_n_const_2(self):
        self.assertEqual([0, 2, 3], sorted(self.rec.top_n_const_2(1, 4)))
        self.assertEqual(0, len(self.rec.top_n_const_2(0, 0)))

    # test top_n_const_3
    # TODO

    # test rec result
    def test_get_top_n(self):
        ret = self.rec.get_top_n()

        cols = [x for x in range(0, self.rec.result_N)]
        cols.extend(['i1', 'i2', 'hl', 'nl'])

        self.assertEqual((4, 14), ret.shape)
        self.assertEqual(cols, list(ret.columns))
        self.assertEqual(4, len(ret))


if __name__ == '__main__':
    unittest.main()

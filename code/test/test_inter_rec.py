import unittest
from sys import path
from surprise import SVD

# setting path
path.append('../src')

from inter_rec import InterRec


class TestPostRec(unittest.TestCase):
    class DummyAlgo(SVD):
        def __init__(self):
            SVD.__init__(self)
            self.i_attr = None
            self.const = None
            self.c_alp = None
            self.c_i1 = None
            self.c_i2 = None
            self.c_nl = None

        def set_data(self, attr, const, c_alp, columns):
            self.i_attr = attr
            self.const = const
            self.c_alp = c_alp
            self.c_i1 = columns[0]
            self.c_i2 = columns[1]
            self.c_nl = columns[2]

    def setUp(self):
        self.dummy = self.DummyAlgo()
        self.rec = InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.dummy)
        self.rec.get_data()
        self.rec.train()
        self.rec.test()

        self.dummy2 = self.DummyAlgo()
        self.rec2 = InterRec('./data/rate2.csv', './data/attr2.csv', './data/const2.csv', self.dummy2)
        self.rec2.get_data()
        self.rec2.train()
        self.rec2.test()

    # load required data
    def test_get_data(self):
        # check algo() object attr
        const = self.dummy.const.loc[self.rec.const.u == 0].iloc[0]
        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], None)

        # check algo() object const
        attr = self.dummy.i_attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 0)

        # check algo() c_alp
        self.assertEqual(self.rec.c_alp, self.dummy.c_alp)

        # check algo() columns
        self.assertEqual(self.rec.c_i1, self.dummy.c_i1)
        self.assertEqual(self.rec.c_i2, self.dummy.c_i2)
        self.assertEqual(self.rec.c_nl, self.dummy.c_nl)

    # sort_prediction()
    def test_sort_predictions(self):
        self.rec.sort_predictions()
        top_n = self.rec.top_N

        self.assertEqual(3, len(top_n[3]))
        self.assertEqual([0, 1, 2], sorted(top_n[2]))

    # sort_prediction slice
    def test_sort_predictions2(self):
        self.rec.result_N = 1
        self.rec.sort_predictions()
        top_n = self.rec.top_N

        self.assertEqual(1, len(top_n[3]))

    # top_n_const_1
    # caution: dummy algorithm doesn't apply constraints
    def test_top_n_const_1(self):
        self.assertEqual([1, 3], sorted(self.rec.top_n_const_1(0, 1)))
        # invalid constraint
        self.assertEqual(0, len(self.rec.top_n_const_1(0, 0)))

    # top_n_const_2
    # caution: dummy algorithm doesn't apply constraints
    def test_top_n_const_2(self):
        self.assertEqual([0, 2, 3], sorted(self.rec.top_n_const_2(1, 4)))
        # invalid constraint
        self.assertEqual(0, len(self.rec.top_n_const_2(0, 0)))

    # top_n_const_3
    # caution: dummy algorithm doesn't apply constraints
    def test_top_n_const_3(self):
        self.assertEqual([0, 1, 2], sorted(self.rec.top_n_const_3(2, [1, 0, 1, 1])))
        # invalid constraint
        self.assertEqual(0, len(self.rec.top_n_const_3(0, [1, 0, 1, 1])))

    # top_n_const
    # caution: dummy algorithm doesn't apply constraints
    def test_top_n_const(self):
        inner_uid = self.rec2.algo.trainset.to_inner_uid
        inner_iid = self.rec2.algo.trainset.to_inner_iid

        pre = self.rec2.top_n_const(6, 0, 1, [1, 1])
        self.assertEqual(6, len(pre))
        for i in range(1, 6):
            e1 = self.rec2.algo.predict(inner_uid('6'), inner_iid(str(pre[i-1])))[3]
            e2 = self.rec2.algo.predict(inner_uid('6'), inner_iid(str(pre[i])))[3]
            self.assertGreaterEqual(e1, e2)

        # invalid constraint
        self.assertEqual(0, len(self.rec2.top_n_const(0, 1, 1, [1, 1])))

    # test rec result
    def test_get_top_n(self):
        ret = self.rec.get_top_n()

        cols = [x for x in range(0, self.rec.result_N)]
        cols.extend(['i1', 'i2', 'nl'])

        self.assertEqual((4, 13), ret.shape)
        self.assertEqual(cols, list(ret.columns))
        self.assertEqual(4, len(ret))

    def test_get_top_n2(self):
        cols = [x for x in range(0, self.rec.result_N)]
        cols.extend(['i1', 'i2', 'nl'])

        ret = self.rec2.get_top_n()
        self.assertEqual((8, 13), ret.shape)
        self.assertEqual(cols, list(ret.columns))
        self.assertEqual(8, len(ret))


if __name__ == '__main__':
    unittest.main()

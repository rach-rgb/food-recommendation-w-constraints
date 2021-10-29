import unittest
from sys import path

# setting path
path.append('../src')

import src.inter_rec as inter_rec
import src.TF_algo as TF_algo


class TestSVDTF(unittest.TestCase):
    def setUp(self):
        self.algo = TF_algo.SVDtf()
        self.rec = inter_rec.InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.algo)
        self.rec.get_data()

    # test set_data
    def test_set_data(self):
        # check algo() object attr
        const = self.algo.const.loc[self.rec.const.u == 0].iloc[0]
        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], "")

        # check algo() object const
        attr = self.algo.i_attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 200)

        # check algo() n_err
        self.assertEqual(self.rec.n_err, self.algo.n_err)

    # test include_ingr
    def test_include_ingr(self):
        self.assertTrue(self.algo.include_ingr(0, 1))
        self.assertFalse(self.algo.include_ingr(0, 4))
        self.assertFalse(self.algo.include_ingr(0, '1'))

    # test exclude_ingr
    def test_exclude_ingr(self):
        self.assertTrue(self.algo.exclude_ingr(0, 4))
        self.assertFalse(self.algo.exclude_ingr(0, 1))
        self.assertTrue(self.algo.exclude_ingr(0, '1'))

    # test satisfy_nutr
    # TODO

    # test check_constraint
    # TODO add constraint 3
    def test_check_constraint(self):
        const1 = self.algo.const.loc[self.rec.const.u == 0].iloc[0]
        const2 = self.algo.const.loc[self.rec.const.u == 1].iloc[0]

        self.assertEqual(0.0, self.algo.check_constraint(0, const1))
        self.assertEqual(1.0, self.algo.check_constraint(2, const1))
        self.assertEqual(0.0, self.algo.check_constraint(0, const2))
        self.assertEqual(1.0, self.algo.check_constraint(2, const2))

    # test fit()
    def test_fit(self):
        self.rec.train()  # this calls self.algo.fit()
        sat = self.algo.vio
        inner_uid = self.algo.trainset.to_inner_uid
        inner_iid = self.algo.trainset.to_inner_iid

        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('0')])
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('0')])
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('1')])
        self.assertEqual(1.0, sat[inner_uid('0')][inner_iid('2')])
        self.assertEqual(0.0, sat[inner_uid('1')][inner_iid('0')])
        self.assertEqual(1.0, sat[inner_uid('1')][inner_iid('2')])

    # test estimate()
    def test_estimate(self):
        self.rec.train()
        inner_uid = self.algo.trainset.to_inner_uid
        inner_iid = self.algo.trainset.to_inner_iid

        self.assertNotEqual(0.0, self.algo.estimate(inner_uid('0'), inner_iid('0')))
        self.assertEqual(0.0, self.algo.estimate(inner_uid('0'), inner_iid('2')))


if __name__ == '__main__':
    unittest.main()

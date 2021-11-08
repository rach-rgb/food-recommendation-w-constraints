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
        self.assertEqual(const['nl'], None)

        # check algo() object const
        attr = self.algo.i_attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 0)

        # check algo() c_alp
        self.assertEqual(self.rec.c_alp, self.algo.c_alp)

        # check algo() column names
        self.assertEqual(self.rec.c_i1, self.algo.c_i1)
        self.assertEqual(self.rec.c_i2, self.algo.c_i2)
        self.assertEqual(self.rec.c_nl, self.algo.c_nl)

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

    # test apply_nutr
    def test_apply_nutr(self):
        weight = self.algo.c_alp

        # FYI
        # self.rec.attr.loc[0].nutrition = [0, 1, 1, 1]
        # self.rec.attr.loc[1].nutrition = [1, 0, 1, 1]
        # self.rec.attr.loc[2].nutrition = [2, 0, 2, 2]
        # self.rec.attr.loc[3].nutrition = [1, 0, 0, 0]

        list1 = [1, 0, 1, 1]
        list2 = [2, 0, 2, 2]
        list3 = [1, 0, 0, 0]

        self.assertEqual(1.667, round(self.algo.apply_nutr(0, list1), 3))
        self.assertEqual(1.667, round(self.algo.apply_nutr(0, list2), 3))
        self.assertEqual(2.500, round(self.algo.apply_nutr(1, list1), 3))
        self.assertEqual(2.500, round(self.algo.apply_nutr(1, list2), 3))
        self.assertEqual(0.000, round(self.algo.apply_nutr(0, list3), 3))

    # test check_constraint
    def test_check_constraint(self):
        const1 = self.algo.const.loc[self.rec.const.u == 0].iloc[0]
        const2 = self.algo.const.loc[self.rec.const.u == 1].iloc[0]
        const3 = self.algo.const.loc[self.rec.const.u == 2].iloc[0]

        self.assertEqual(0.0, self.algo.check_constraint(0, const1))  # satisfy
        self.assertEqual(-1.0, self.algo.check_constraint(2, const1))  # not satisfy
        self.assertEqual(0.0, self.algo.check_constraint(0, const2))  # satisfy
        self.assertEqual(-1.0, self.algo.check_constraint(2, const2))  # not satisfy
        self.assertEqual(2.667, round(self.algo.check_constraint(0, const3), 3))  # score = 1.667
        self.assertEqual(3.500, round(self.algo.check_constraint(1, const3), 3))  # score = 2.500

    # test fit()
    def test_fit(self):
        self.rec.train()  # this calls self.algo.fit()
        sat = self.algo.vio
        inner_uid = self.algo.trainset.to_inner_uid
        inner_iid = self.algo.trainset.to_inner_iid

        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('0')])  # satisfy
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('0')])  # satisfy
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('1')])  # satisfy
        self.assertEqual(-1.0, sat[inner_uid('0')][inner_iid('2')])  # not satisfy
        self.assertEqual(0.0, sat[inner_uid('1')][inner_iid('0')])  # satisfy
        self.assertEqual(-1.0, sat[inner_uid('1')][inner_iid('2')])  # not satisfy
        self.assertEqual(2.667, round(sat[inner_uid('2')][inner_iid('0')], 3))  # score = 1.667
        self.assertEqual(3.500, round(sat[inner_uid('2')][inner_iid('1')], 3))  # score = 2.500
        self.assertEqual(3.500, round(sat[inner_uid('2')][inner_iid('2')], 3))  # score = 2.500

    # test estimate()
    def test_estimate(self):
        self.rec.train()
        inner_uid = self.algo.trainset.to_inner_uid
        inner_iid = self.algo.trainset.to_inner_iid

        self.assertNotEqual(0.0, self.algo.estimate(inner_uid('0'), inner_iid('0')))  # violate constraint
        self.assertEqual(0.0, self.algo.estimate(inner_uid('0'), inner_iid('2')))  # violate constraint
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('0')) >= 1.667)
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('0')) <= 2.5 + 1.667)
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('1')) >= 2.5)
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('1')) <= 2.5 + 1.667)

if __name__ == '__main__':
    unittest.main()

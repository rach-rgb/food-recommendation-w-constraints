import unittest
import numpy as np
from sys import path

# setting path
path.append('../src')

from inter_rec import InterRec
import svd_constraint


class TestCnstSVD(unittest.TestCase):
    def setUp(self):
        self.algo = svd_constraint.CnstSVD()
        self.rec = InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.algo)
        self.rec.get_data()

        self.algo2 = svd_constraint.CnstSVD()
        self.rec2 = InterRec('./data/rate2.csv', './data/attr2.csv', './data/const2.csv', self.algo2)
        self.rec2.get_data()

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
        def round_apply_nutr(id, target):
            return round(self.algo.apply_nutr(id, target), 3)

        self.assertEqual(1.667, round_apply_nutr(0, [1, 0, 1, 1]))
        self.assertEqual(1.667, round_apply_nutr(0, [2, 0, 2, 2]))
        self.assertEqual(2.500, round_apply_nutr(1, [1, 0, 1, 1]))
        self.assertEqual(2.500, round_apply_nutr(1, [2, 0, 2, 2]))
        self.assertEqual(0.000, round_apply_nutr(0, [1, 0, 0, 0]))

    # check_constraint() for single constraint
    def test_check_constraint(self):
        def get_const(u):
            return self.algo.const.loc[self.rec.const.u == u].iloc[0]

        const0 = get_const(0)
        const1 = get_const(1)
        const2 = get_const(2)

        self.assertEqual(0.0, self.algo.check_constraint(0, const0))  # satisfy
        self.assertEqual(-1.0, self.algo.check_constraint(2, const0))  # not satisfy
        self.assertEqual(0.0, self.algo.check_constraint(0, const1))  # satisfy
        self.assertEqual(-1.0, self.algo.check_constraint(2, const1))  # not satisfy
        self.assertEqual(2.667, round(self.algo.check_constraint(0, const2), 3))  # score = 1.667
        self.assertEqual(3.500, round(self.algo.check_constraint(1, const2), 3))  # score = 2.500

    # check_constraint for mixed constraint
    def test_check_constraint2(self):
        def get_const(u):
            return self.algo2.const.loc[self.rec2.const.u == u].iloc[0]

        const0 = get_const(0)  # constraint 1
        const3 = get_const(3)  # constraint 1 & 2
        const4 = get_const(4)  # constraint 2 & 3
        const6 = get_const(6)  # constraint 1 & 2 & 3

        self.assertEqual(0.0, self.algo2.check_constraint(0, const0))  # satisfy
        self.assertEqual(-1.0, self.algo2.check_constraint(3, const0))  # not satisfy
        self.assertEqual(0.0, self.algo2.check_constraint(0, const3))  # satisfy both
        self.assertEqual(-1.0, self.algo2.check_constraint(1, const3))  # violates constraint 2
        self.assertEqual(-1.0, self.algo2.check_constraint(1, const4))  # violates constraint 2
        score1 = self.algo2.check_constraint(0, const4)  # score = 2.5
        score2 = self.algo2.check_constraint(2, const4)  # score = 0.0
        self.assertLessEqual(1.0, score2)  # satisfy constraint 2 & score: 0 < 2
        self.assertLessEqual(score2, score1)
        self.assertLessEqual(score1, 6.0)
        self.assertEqual(3.500, round(self.algo2.check_constraint(0, const6), 3))  # score = 2.500
        self.assertEqual(-1.0, self.algo2.check_constraint(3, const6)) # violates constraint 2

    # fit() for single constraint
    def test_fit(self):
        self.rec.train()  # this calls self.algo.fit()
        sat = self.algo.vio
        inner_uid = self.algo.trainset.to_inner_uid
        inner_iid = self.algo.trainset.to_inner_iid

        self.assertEqual(0.0, sat[inner_uid('3')][inner_iid('0')])  # no constraint
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('0')])  # satisfy
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('0')])  # satisfy
        self.assertEqual(0.0, sat[inner_uid('0')][inner_iid('1')])  # satisfy
        self.assertEqual(-1.0, sat[inner_uid('0')][inner_iid('2')])  # not satisfy
        self.assertEqual(0.0, sat[inner_uid('1')][inner_iid('0')])  # satisfy
        self.assertEqual(-1.0, sat[inner_uid('1')][inner_iid('2')])  # not satisfy
        self.assertEqual(2.667, round(sat[inner_uid('2')][inner_iid('0')], 3))  # score = 1.667
        self.assertEqual(3.500, round(sat[inner_uid('2')][inner_iid('1')], 3))  # score = 2.500
        self.assertEqual(3.500, round(sat[inner_uid('2')][inner_iid('2')], 3))  # score = 2.500

    # fit() for mixed constraint
    def test_fit2(self):
        self.rec2.train()
        sat = self.algo2.vio
        inner_uid = self.algo2.trainset.to_inner_uid
        inner_iid = self.algo2.trainset.to_inner_iid

        # violate matrix: 9 users * 7 items
        m = [
            [0, 0, 0, -1, -1, -1, 0],
            [0, -1, 0, 0, -1, 0, 0],
            [3.5, 3.5, 1.0, 3.5, 3.5, 1.0, 3.5],
            [0, -1, 0, -1, -1, -1, 0],
            [3.5, -1, 1.0, 3.5, -1, 1.0, 3.5],
            [3.5, 3.5, 1.0, -1, -1, -1, 3.5],
            [3.5, -1, 1.0, -1, -1, -1, 3.5],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]
        ]

        for u in range(0, 9):
            for i in range(0, 7):
                self.assertEqual(m[u][i], round(sat[inner_uid(str(u))][inner_iid(str(i))], 2))

    # estimate() for single constraint
    def test_estimate(self):
        self.rec.train()
        inner_uid = self.algo.trainset.to_inner_uid
        inner_iid = self.algo.trainset.to_inner_iid

        u = inner_uid('0')
        i = inner_iid('0')
        est = self.algo.trainset.global_mean + self.algo.bu[u] + self.algo.bi[i] \
              + np.dot(self.algo.qi[i], self.algo.pu[u])  # satisfy constraint
        self.assertEqual(self.algo.estimate(inner_uid('0'), inner_iid('0')), est)
        self.assertEqual(0.0, self.algo.estimate(inner_uid('0'), inner_iid('2')))  # violate constraint

        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('0')) >= 1.667)
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('0')) <= 2.5 + 1.667)
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('1')) >= 2.5)
        self.assertTrue(self.algo.estimate(inner_uid('2'), inner_iid('1')) <= 2.5 + 1.667)

        u = inner_uid('3')
        i = inner_iid('0')
        est = self.algo.trainset.global_mean + self.algo.bu[u] + self.algo.bi[i] \
              + np.dot(self.algo.qi[i], self.algo.pu[u])
        # follow original SVD method if there's no constraint
        self.assertEqual(self.algo.estimate(inner_uid('3'), inner_iid('0')), est)

    # estimate() for mixed constraint
    def test_estimate2(self):
        self.rec2.train()
        inner_uid = self.algo2.trainset.to_inner_uid
        inner_iid = self.algo2.trainset.to_inner_iid

        u = inner_uid('2')
        i = inner_iid('2')
        est = self.algo2.trainset.global_mean + self.algo2.bu[u] + self.algo2.bi[i] \
              + np.dot(self.algo2.qi[i], self.algo2.pu[u])  # satisfy constraint
        self.assertGreaterEqual(est * (1 - self.algo2.c_alp), self.algo2.estimate(inner_uid('2'), inner_iid('2')))

        self.assertLessEqual(2.5, self.algo2.estimate(inner_uid('2'), inner_iid('0')))
        self.assertGreaterEqual(self.algo2.estimate(inner_uid('6'), inner_iid('0')),
                                self.algo2.estimate(inner_uid('6'), inner_iid('2')))
        self.assertEqual(0.0, self.algo2.estimate(inner_uid('6'), inner_iid('3')))

    # exclude_train() for CnstSVD
    def test_exclude_train(self):
        self.rec.train()

        # change values for test
        self.algo.vio[0][0] = -1.0
        self.algo.vio[0][1] = 0.0
        self.algo.vio[0][2] = 1.0
        self.algo.vio[1][0] = 3.5
        self.algo.vio[1][1] = 6.0

        self.assertTrue(self.algo.exclude_train(0, 0))
        # include train
        self.assertFalse(self.algo.exclude_train(0, 1))
        self.assertTrue(self.algo.exclude_train(0, 2))
        self.assertTrue(self.algo.exclude_train(1, 0))
        self.assertTrue(self.algo.exclude_train(1, 1))


class TestCnstSVD_all(unittest.TestCase):
    def setUp(self):
        self.algo = svd_constraint.CnstSVD_all()
        self.rec = InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.algo)
        self.rec.get_data()
        self.rec.train()

    def test_exclude_train(self):
        # change values for test
        self.algo.vio[0][0] = -1.0
        self.algo.vio[0][1] = 0.0
        self.algo.vio[0][2] = 1.0
        self.algo.vio[1][0] = 3.5
        self.algo.vio[1][1] = 6.0

        self.assertFalse(self.algo.exclude_train(0, 0))
        self.assertFalse(self.algo.exclude_train(0, 1))
        self.assertFalse(self.algo.exclude_train(0, 2))
        self.assertFalse(self.algo.exclude_train(1, 0))
        self.assertFalse(self.algo.exclude_train(1, 1))


class TestCnstSVD_weaker(unittest.TestCase):
    def setUp(self):
        self.algo = svd_constraint.CnstSVD_weaker()
        self.rec = InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.algo)
        self.rec.get_data()
        self.rec.train()

    def test_exclude_train(self):
        # change values for test
        self.algo.vio[0][0] = -1.0
        self.algo.vio[0][1] = 0.0
        self.algo.vio[0][2] = 1.0
        self.algo.vio[1][0] = 3.5
        self.algo.vio[1][1] = 6.0

        self.assertTrue(self.algo.exclude_train(0, 0))
        self.assertFalse(self.algo.exclude_train(0, 1))
        self.assertFalse(self.algo.exclude_train(0, 2))
        self.assertFalse(self.algo.exclude_train(1, 0))
        self.assertFalse(self.algo.exclude_train(1, 1))


class TestCnstSVD_weak(unittest.TestCase):
    def setUp(self):
        self.algo = svd_constraint.CnstSVD_weak()
        self.rec = InterRec('./data/rate.csv', './data/attr.csv', './data/const.csv', self.algo)
        self.rec.get_data()
        self.rec.train()

    def test_exclude_train(self):
        # change values for test
        self.algo.vio[0][0] = -1.0
        self.algo.vio[0][1] = 0.0
        self.algo.vio[0][2] = 1.0
        self.algo.vio[1][0] = 3.5
        self.algo.vio[1][1] = 6.0

        self.assertTrue(self.algo.exclude_train(0, 0))
        self.assertFalse(self.algo.exclude_train(0, 1))
        self.assertTrue(self.algo.exclude_train(0, 2))
        self.assertFalse(self.algo.exclude_train(1, 0))
        self.assertFalse(self.algo.exclude_train(1, 1))


if __name__ == '__main__':
    unittest.main()

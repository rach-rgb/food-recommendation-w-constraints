import unittest
from sys import path
from collections import defaultdict
from surprise import accuracy

# setting path
path.append('../src')

from post_rec import PostRec


class TestPostRec(unittest.TestCase):
    def setUp(self):
        self.rec = PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv')
        self.rec.get_data()
        self.rec.train()
        self.rec.test()

        self.rec_split = PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv', need_test=True)
        self.rec_split.get_data()
        self.rec_split.train()
        self.rec_split.test()

        self.rec2 = PostRec('./data/rate2.csv', './data/attr2.csv', './data/const2.csv')
        self.rec2.get_data()
        self.rec2.train()
        self.rec2.test()

        self.rec2_split = PostRec('./data/rate2.csv', './data/attr2.csv', './data/const2.csv', need_test=True)
        self.rec2_split.get_data()
        self.rec2_split.train()
        self.rec2_split.test()

    # test()
    def test_test(self):
        self.rec.test()
        result = defaultdict(list)
        for uid, iid, true_r, est, _ in self.rec.predictions:
            result[int(uid)].append(int(iid))

        # prediction w/o constraints
        self.assertEqual([1, 3], sorted(result[0]))
        self.assertEqual([0, 2, 3], sorted(result[1]))
        self.assertEqual([0, 1, 2], sorted(result[2]))
        self.assertTrue([0, 2, 3], sorted(result[3]))

    # test_rmse() for single constraint
    def test_rmse1(self):
        def predict_by_algo(u, i):
            return self.rec_split.algo.predict(u, i)[3]

        self.rec_split.test_set = [('0', '2', 5.0), ('1', '2', 5.0)]
        # Post-RS predicts rate as 0 if constraint is violated
        self.assertEqual(5.0, accuracy.rmse(self.rec_split.test_rmse(), False))

        self.rec_split.test_set = [('0', '0', 4.0)]
        # Post-RS follows rating from algorithm if constraint is satisfied
        self.assertEqual(predict_by_algo('0', '0'), self.rec_split.test_rmse()[0][3])

        self.rec_split.test_set = [('3', '1', 5.0)]
        # Post-RS follows rating from algorithm if there's no constraint
        self.assertEqual(predict_by_algo('3', '1'), self.rec_split.test_rmse()[0][3])

        self.rec_split.test_set = [('2', '1', 5.0)]
        # Post-RS applies nutrient score
        predictions = self.rec_split.test_rmse()
        est_algo = predict_by_algo('2', '1') * (1 - self.rec_split.c_alp) + 5.0 * self.rec_split.c_alp
        self.assertEqual(round(est_algo, 5), round(predictions[0][3], 5))

        self.rec_split.test_set = [('2', '0', 5.0)]
        # Post-RS applies nutrient score
        predictions = self.rec_split.test_rmse()
        est_algo = predict_by_algo('2', '0') * (1 - self.rec_split.c_alp) \
                   + 5 * self.rec_split.c_alp * self.rec_split.apply_nutr(0, [1, 0, 1, 1])
        self.assertEqual(round(est_algo, 5), round(predictions[0][3], 5))

    # test_rmse() for mixed constraint
    def test_rmse2(self):
        def predict_by_algo(u, i):
            return self.rec2_split.algo.predict(u, i)[3]

        self.rec2_split.test_set = [('0', '0', 3), ('3', '1', 3), ('3', '3', 3), ('4', '0', 4),
                                    ('6', '2', 3), ('6', '3', 4)]
        pre_by_algo = self.rec2_split.algo.test(self.rec2_split.test_set)
        pre_by_rs = self.rec2_split.test_rmse()

        self.assertEqual(pre_by_algo[0][3], pre_by_rs[0][3])
        self.assertNotEqual(pre_by_algo[1][3], pre_by_rs[1][3])
        self.assertEqual(0, pre_by_rs[1][3])
        self.assertEqual(0, pre_by_rs[2][3])
        est_algo = predict_by_algo('4', '0') * (1 - self.rec2_split.c_alp) + 5.0 * self.rec2_split.c_alp
        self.assertEqual(round(est_algo, 5), round(pre_by_rs[3][3], 5))
        self.assertLess(0, pre_by_rs[4][3])
        self.assertEqual(0, pre_by_rs[5][3])

    # sort_prediction()
    def test_sort_prediction(self):
        self.rec.sort_prediction()

        for j in range(0, 4):
            sorted_top_K = self.rec.top_K[j]
            for i in range(1, len(sorted_top_K)):
                self.assertGreaterEqual(sorted_top_K[i-1][1], sorted_top_K[i][1])

    # top_n_const_1()
    def test_top_n_const_1(self):
        result = self.rec.top_n_const_1(0, 1)
        attr1 = self.rec.attr.loc[1]
        attr2 = self.rec.attr.loc[2]
        attr3 = self.rec.attr.loc[3]

        self.assertEqual([1, 3], sorted(result))
        self.assertIn(1, attr1.ingredient_ids)
        self.assertNotIn(1, attr2.ingredient_ids)
        self.assertIn(1, attr3.ingredient_ids)
        # invalid constraint
        self.assertEqual(0, len(self.rec.top_n_const_1(0, 2)))

    # top_n_const_2()
    def test_top_n_const_2(self):
        result = self.rec.top_n_const_2(1, 4)
        attr0 = self.rec.attr.loc[0]

        self.assertEqual([0], sorted(result))
        self.assertNotIn(4, attr0.ingredient_ids)
        # invalid constraint
        self.assertEqual(0, len(self.rec.top_n_const_2(1, 3)))

    # top_n_const_3()
    def test_top_n_const_3(self):
        result = self.rec.top_n_const_3(2, [1, 0, 1, 1])

        self.assertEqual([0, 1, 2], sorted(result))

        # modify parameters for test
        self.rec.c_alp = 1
        result = self.rec.top_n_const_3(2, [1, 0, 1, 1])
        self.assertEqual([0, 1, 2], sorted(result))
        self.assertEqual(0, result[2])  # score of item 0 is 0, score of item 1, 2 is 1

    # top_n_const_3() slice results
    def test_top_n_const_3_2(self):
        # modify parameters for testing
        self.rec.result_N = 1
        self.rec.c_alp = 1
        result = self.rec.top_n_const_3(2, [1, 0, 1, 1])

        self.assertTrue((result[0] == 1) or (result[0] == 2))

    # top_n_const()
    def test_top_n_const(self):
        # no constraint
        self.assertEqual([0, 1, 2, 3, 4, 5], sorted(self.rec2.top_n_const(7)))
        # invalid constraint
        self.assertEqual([], self.rec2.top_n_const(0, 1, 1, 1))
        # mixe constraint
        self.assertEqual([0, 1, 2], sorted(self.rec2.top_n_const(0, 0)))
        self.assertEqual([0, 2, 3, 5], sorted(self.rec2.top_n_const(1, iid2=1)))
        self.assertEqual([0, 1, 3, 4], sorted(self.rec2.top_n_const(2, target=[1, 1])[:4]))
        self.assertEqual([2, 5], sorted(self.rec2.top_n_const(2, target=[1, 1])[4:]))
        self.assertEqual([0, 2], sorted(self.rec2.top_n_const(3, 0, 1)))
        self.assertEqual([0, 3], sorted(self.rec2.top_n_const(4, iid2=1, target=[1, 1])[:2]))
        self.assertEqual([2, 5], sorted(self.rec2.top_n_const(4, iid2=1, target=[1, 1])[2:]))
        self.assertEqual([0, 1], sorted(self.rec2.top_n_const(5, 0, target=[1, 1])[:2]))
        self.assertEqual([2], sorted(self.rec2.top_n_const(5, 0, target=[1, 1])[2:]))
        self.assertEqual([0], sorted(self.rec2.top_n_const(6, 0, 1, [1, 1])[:1]))
        self.assertEqual([2], sorted(self.rec2.top_n_const(6, 0, 1, [1, 1])[1:]))

    # get_top_n() for single constraint
    def test_get_top_n(self):
        ret = self.rec.get_top_n()

        cols = [x for x in range(0, self.rec.result_N)]
        cols.extend(['i1', 'i2', 'nl'])

        self.assertEqual((4, 13), ret.shape)
        self.assertEqual(cols, list(ret.columns))
        self.assertEqual(4, len(ret))

    # get_top_n() for mixed constraint
    def test_get_top_n2(self):
        cols = [x for x in range(0, self.rec2.result_N)]
        cols.extend(['i1', 'i2', 'nl'])

        ret = self.rec2.get_top_n()
        self.assertEqual((8, 13), ret.shape)
        self.assertEqual(cols, list(ret.columns))


if __name__ == '__main__':
    unittest.main()

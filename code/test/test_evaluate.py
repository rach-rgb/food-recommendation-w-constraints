from sys import path
import unittest, math
from collections import defaultdict
import pandas as pd
from surprise import SVD

# setting path
path.append('../src')

from evaluate import Evaluation as ev
from rec_base import FoodRecBase


class TestEval(unittest.TestCase):
    # Dummy Recommendation System for testing
    class DummyRS(FoodRecBase):
        def top_n_const_1(self, uid, iid):
            return []

        def top_n_const_2(self, uid, iid):
            return []

        def top_n_const_3(self, uid, target):
            return []

        def top_n_const(self, uid, iid1, iid2, target):
            return []

        def get_top_n(self):
            top_N = defaultdict(list)
            for (u, i, r) in self.test_set:
                top_N[int(u)].append(i)

            result = pd.DataFrame.from_dict(top_N, orient='index')
            result = result.reindex(columns=[x for x in range(0, self.result_N)])
            return result.join(self.const.set_index('u'))

    def setUp(self):
        self.rec = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), need_test=True)
        self.rec.get_data()
        self.rec.train()

    # calculate_rmse()
    def test_calculate_rmse(self):
        predictions = self.rec.test_rmse()

        self.assertGreater(5.0, ev.calculate_rmse(predictions))
        self.assertLessEqual(0.0, ev.calculate_rmse(predictions))

    # cal_ndcg()
    # len(gt) = k
    def test_cal_ndcg(self):
        gt = [1, 2, 3, 4, 5]
        res1 = [1, 2, 3, 4, 5]  # perfect result
        res2 = [5, 4, 3, 2, 1]
        res3 = [6, 7, 8, 9, 10]
        res4 = [6, 7, 3, 2, 1]
        res5 = [3, 2, 1, 6, 7]

        self.assertAlmostEqual(1.0, ev.cal_ndcg(gt, res1))
        self.assertAlmostEqual(1.0, ev.cal_ndcg(gt, res2))
        # relevance of item in gt is all 1
        self.assertAlmostEqual(ev.cal_ndcg(gt, res1), ev.cal_ndcg(gt, res2))

        self.assertAlmostEqual(0.0, ev.cal_ndcg(gt, res3))

        # res5 has item in gt at higher rank
        self.assertGreater(ev.cal_ndcg(gt, res5), ev.cal_ndcg(gt, res4))

    # cal_ndcg() when len(gt) != k
    def test_cal_ndcg2(self):
        gt = [1, 2, 3, 4, 5]

        res1 = [1, 2, 3, 4, 5]
        res2 = [1, 2, 3]
        res3 = [4, 5, 6]
        res4 = [1, 2, 3, 4, 5, 6]
        res5 = [6, 1, 2, 3, 4, 5]
        res6 = [1, 2, 3, 4, 5, None]

        # len(gt) = k
        self.assertAlmostEqual(1.0, ev.cal_ndcg(gt, res1))
        # len(gt) > k
        self.assertAlmostEqual(1.0, ev.cal_ndcg(gt, res2))
        self.assertGreater(1.0, ev.cal_ndcg(gt, res3))
        # len(gt) < k
        self.assertAlmostEqual(1.0, ev.cal_ndcg(gt, res4))
        self.assertGreater(1.0, ev.cal_ndcg(gt, res5))
        # contains None
        self.assertAlmostEqual(1.0, ev.cal_ndcg(gt, res6))

    # calculate_ndcg()
    def test_calculate_ndcg(self):
        rel_dict = {0: ['A', 'B', 'C', 'D', 'E']}
        top_n_df = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E']}).transpose()

        self.assertAlmostEqual(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 5))
        self.assertAlmostEqual(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 3))
        self.assertAlmostEqual(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 1))

        rel_dict = {0: ['A', 'B', 'C', 'D', 'E']}
        top_n_df = pd.DataFrame({0: ['F', 'G', 'C', 'D', 'E']}).transpose()

        self.assertGreater(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 5))
        self.assertLess(ev.calculate_ndcg(rel_dict, top_n_df, 3), ev.calculate_ndcg(rel_dict, top_n_df, 5))
        self.assertAlmostEqual(0.0, ev.calculate_ndcg(rel_dict, top_n_df, 1))

    # calculate_ndcg() for multiple users
    def test_calculate_ndcg2(self):
        rel_dict = {0: ['A', 'B', 'C', 'D', 'E'], 3: ['A', 'B', 'C', 'D', 'E']}
        top_n_df = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E'], 3: ['F', 'G', 'C', 'D', 'E']}).transpose()

        self.assertGreater(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 5))
        self.assertAlmostEqual(0.5, ev.calculate_ndcg(rel_dict, top_n_df, 2))

    # skip either rel_dict[u] is empty or top_n_df.loc[u] is empty
    def test_calculate_ndcg3(self):
        rel_dict = {0: ['A', 'B', 'C', 'D', 'E']}
        top_n_df = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E'], 3: ['F', 'G', 'C', 'D', 'E']}).transpose()

        self.assertAlmostEqual(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 5))

        rel_dict = {0: ['A', 'B', 'C', 'D', 'E'], 4: ['A', 'B', 'C', 'D', 'E']}
        top_n_df = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E'], 3: ['F', 'G', 'C', 'D', 'E']}).transpose()

        self.assertAlmostEqual(1.0, ev.calculate_ndcg(rel_dict, top_n_df, 5))

    # return nan for empty rel_dict
    def test_calculate_ndcg4(self):
        rel_dict = {}
        top_n_df = self.rec.get_top_n()

        self.assertTrue(math.isnan(ev.calculate_ndcg(rel_dict, top_n_df, 5)))

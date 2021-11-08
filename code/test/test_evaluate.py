import unittest
import math
from sys import path
import pandas as pd

# setting path
path.append('../src')

import src.evaluate as ev
import src.post_rec as post_rec
import src.load_data as ld


class TestEvaluate(unittest.TestCase):

    # test constructing relevancy list
    def test_construct(self):
        ev1 = ev.NDCG()

        self.assertEqual(10, ev1.alen)
        self.assertEqual([3, 3, 3, 2, 2, 2, 1, 1, 1, 1], ev1.rel)

        ev2 = ev.NDCG(3, 3)
        self.assertEqual([3, 2, 1], ev2.rel)

    # test cal_idcg
    def test_cal_idcg(self):
        ev1 = ev.NDCG(5, 5)
        val1 = 3 / math.log(2, 2) + 2 / math.log(3, 2) + 1 / math.log(4, 2) + 1 / math.log(5, 2) + 1 / math.log(6, 2)
        self.assertEqual(val1, ev1.cal_idcg(5))

        ev2 = ev.NDCG(10, 10)
        val2 = 3 / math.log(2, 2) + 3 / math.log(3, 2) + 3 / math.log(4, 2)
        val2 = val2 + 2 / math.log(5, 2) + 2 / math.log(6, 2) + 2 / math.log(7, 2)
        val2 = val2 + 1 / math.log(8, 2) + 1 / math.log(9, 2) + 1 / math.log(10, 2) + 1 / math.log(11, 2)
        self.assertEqual(val2, ev2.cal_idcg(10))

    # test cal_ndcg
    def test_cal_ndcg(self):
        ev1 = ev.NDCG(5, 5)

        answ1 = [1, 2, 3, 4, 5]
        res1 = [1, 2, 3, 4, 5]  # perfect result
        res2 = [5, 4, 3, 2, 1]
        res3 = [6, 7, 8, 9, 10]

        self.assertEqual(1.0, ev1.cal_ndcg(answ1, res1))
        self.assertEqual(0.0, ev1.cal_ndcg(answ1, res3))
        self.assertGreater(ev1.cal_ndcg(answ1, res1), ev1.cal_ndcg(answ1, res2))

    # cal_ndcg for len(answ) != len(res)
    def test_cal_ndcg2(self):
        ev1 = ev.NDCG(6, 5)
        ev1.rel = [3, 3, 2, 2, 1, 1]

        answ = ['A', 'B', 'C', 'D', 'E', 'F']
        res1 = ['A', 'E', 'C', 'D', 'F']
        res2 = ['A', 'B', 'C', 'G', 'E']

        self.assertEqual(7.141, round(ev1.cal_idcg(5), 3))
        self.assertEqual(0.823, round(ev1.cal_ndcg(answ, res1), 3))
        self.assertEqual(0.879, round(ev1.cal_ndcg(answ, res2), 3))

    # cal_ndcg if result contains None
    def test_cal_ndcg3(self):
        ev1 = ev.NDCG(5, 5)

        answ = ['A', 'B', 'C', 'D', 'E']
        res1 = ['A', 'B', 'C', None, None]
        res2 = ['A', 'B', 'C', 'D', 'E']

        self.assertGreater(ev1.cal_ndcg(answ, res2), ev1.cal_ndcg(answ, res1))

    # test avg_ndcg
    def test_avg_ndcg(self):
        ev1 = ev.NDCG(5, 5)

        adf = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E']}).transpose()
        rdf = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E']}).transpose()
        self.assertEqual(1.0, ev1.avg_ndcg(adf, rdf))

        adf = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E'], 1: ['A', 'B', 'C', 'D', 'E']}).transpose()
        rdf = pd.DataFrame({0: ['A', 'B', 'C', 'D', 'E'], 1: ['F', 'F', 'F', 'F', 'F']}).transpose()
        self.assertEqual(0.5, ev1.avg_ndcg(adf, rdf))

    # avg_ndcg for len(answ) != len(result)
    def test_avg_ndcg2(self):
        ev1 = ev.NDCG(5, 3)

        adf = pd.DataFrame({'0': ['A', 'B', 'C', 'D', 'E'], '1': ['A', 'B', 'C', 'D', 'E']}).transpose()
        rdf = pd.DataFrame({'0': ['A', 'B', 'C'], '1': ['F', 'F', 'F']}).transpose()

        ndcg = ev1.cal_ndcg(['A', 'B', 'C', 'D', 'E'], ['A', 'B', 'C']) / 2.0

        self.assertEqual(ndcg, ev1.avg_ndcg(adf, rdf))

    # avg_ndcg test
    def test_avg_ndcg3(self):
        ev1 = ev.NDCG(5, 3)

        adf = pd.DataFrame({'0': ['A', 'B', 'C', 'D', 'E'], '1': ['A', 'B', 'C', 'D', 'E'],
                            '2': ['A', 'B', 'C', 'D', 'E']}).transpose()
        rdf = pd.DataFrame({'0': ['A', 'B', 'C'], '1': ['F', 'F', 'F']}).transpose()

        ndcg = ev1.cal_ndcg(['A', 'B', 'C', 'D', 'E'], ['A', 'B', 'C']) / 2.0

        self.assertEqual(ndcg, ev1.avg_ndcg(adf, rdf))

    # test save_rates in load_data.py
    def test_save_rates(self):
        rec2 = post_rec.PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv', split=True)
        rec2.test_RMSE_set = [('0', '0', 5.0), ('0', '1', 4.0), ('0', '2', 3.0), ('1', '0', 5.0), ('3', '1', 4.0)]

        ld.save_rates(rec2.test_RMSE_set, 'test')
        df = ld.load_rate_df('../../result/test.csv')

        list0 = df.loc[0].iloc[0]
        list1 = df.loc[1].iloc[0]
        list3 = df.loc[3].iloc[0]

        self.assertEqual([0, 1], sorted(list0))
        self.assertEqual([0], sorted(list1))
        self.assertEqual([1], sorted(list3))

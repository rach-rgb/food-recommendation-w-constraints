import unittest
from collections import defaultdict
from surprise import accuracy
from sys import path

# setting path
path.append('../src')

import src.post_rec as post_rec


class TestPostRec(unittest.TestCase):

    def setUp(self):
        self.rec = post_rec.PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv')
        self.rec.get_data()
        self.rec.train()
        self.rec.test()

    # load required data
    def test_get_data(self):
        # check const data
        const = self.rec.const.loc[self.rec.const.u == 0].iloc[0]
        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], "")

        # check attr data
        attr = self.rec.attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 200)

    # split input dataset
    def test_get_data2(self):
        rec2 = post_rec.PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv', split=True)
        rec2.get_data()

    # test result
    def test_test(self):
        result = defaultdict(list)
        for uid, iid, true_r, est, _ in self.rec.predictions:
            result[int(uid)].append(int(iid))

        # check prediction contains all tuples
        self.assertTrue(2, len(result[0]))
        self.assertIn(1, result[0])
        self.assertNotIn(0, result[0])
        self.assertTrue(3, len(result[1]))

    # test_sort_prediction method
    def test_sort_prediction(self):
        self.rec.sort_prediction()

        # check sorted list
        self.assertEqual(sorted(self.rec.top_K[0]), self.rec.top_K[0])
        self.assertEqual(3, len(self.rec.top_K[3]))

    # test include_ingr
    def test_include_ingr(self):
        self.assertTrue(self.rec.include_ingr(0, 1))
        self.assertFalse(self.rec.include_ingr(0, 4))
        self.assertFalse(self.rec.include_ingr(0, '1'))

    # test exclude_ingr
    def test_exclude_ingr(self):
        self.assertTrue(self.rec.exclude_ingr(0, 4))
        self.assertFalse(self.rec.exclude_ingr(0, 1))
        self.assertTrue(self.rec.exclude_ingr(0, '1'))

    # test satisfy_nutr
    # TODO

    # test valid_constraint
    def test_valid_constraint(self):
        self.assertTrue(self.rec.valid_constraint(0, i1=1))
        self.assertTrue(self.rec.valid_constraint(1, i2=4))
        self.assertFalse(self.rec.valid_constraint(2))
        self.assertFalse(self.rec.valid_constraint(0, i1=1, i2=2))

        # TODO

    # test top_n_const_1
    def test_top_n_const_1(self):
        result = self.rec.top_n_const_1(0, 1)
        attr1 = self.rec.attr.loc[1]
        attr3 = self.rec.attr.loc[3]

        self.assertEqual([1, 3], sorted(result))
        self.assertIn(1, attr1.ingredient_ids)
        self.assertIn(1, attr3.ingredient_ids)

        self.assertEqual(0, len(self.rec.top_n_const_1(0, 2)))

    # test top_n_const_2
    def test_top_n_const_2(self):
        result = self.rec.top_n_const_2(1, 4)
        attr0 = self.rec.attr.loc[0]

        self.assertEqual([0], sorted(result))
        self.assertNotIn(4, attr0.ingredient_ids)

        self.assertEqual(0, len(self.rec.top_n_const_2(1, 3)))

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

    # evaluation using RMSE
    def test_RMSE(self):
        rec2 = post_rec.PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv', split=True)
        rec2.get_data()
        rec2.train()
        predictions = rec2.test_rmse()
        accuracy.rmse(predictions)


if __name__ == '__main__':
    unittest.main()

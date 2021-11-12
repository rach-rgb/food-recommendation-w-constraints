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
        self.assertEqual(const['nl'], None)

        # check attr data
        attr = self.rec.attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 0)

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

    # test apply_nutr
    def test_apply_nutr(self):
        # FYI
        # self.rec.attr.loc[0].nutrition = [0, 1, 1, 1]
        # self.rec.attr.loc[1].nutrition = [1, 0, 1, 1]
        # self.rec.attr.loc[2].nutrition = [2, 0, 2, 2]
        # self.rec.attr.loc[3].nutrition = [1, 0, 0, 0]

        list1 = [1, 0, 1, 1]
        list2 = [2, 0, 2, 2]
        list3 = [1, 0, 0, 0]

        self.assertEqual(0.67, round(self.rec.apply_nutr(0, list1), 2))
        self.assertEqual(0.67, round(self.rec.apply_nutr(0, list2), 2))
        self.assertEqual(1.00, round(self.rec.apply_nutr(1, list1), 2))
        self.assertEqual(1.00, round(self.rec.apply_nutr(1, list2), 2))
        self.assertEqual(0.00, round(self.rec.apply_nutr(0, list3), 2))

    # test valid_constraint
    def test_valid_constraint(self):
        self.assertTrue(self.rec.valid_constraint(0, i1=1))
        self.assertTrue(self.rec.valid_constraint(1, i2=4))
        self.assertTrue(self.rec.valid_constraint(2, nl=[1, 0, 1, 1]))

        self.assertFalse(self.rec.valid_constraint(0, i1=1, i2=2))
        self.assertFalse(self.rec.valid_constraint(2, nl=[1, 0, 1, 0.5]))
        self.assertFalse(self.rec.valid_constraint(3))  # No constraint for user 3

    # test top_n_const_1
    def test_top_n_const_1(self):
        result = self.rec.top_n_const_1(0, 1)
        attr1 = self.rec.attr.loc[1]
        attr2 = self.rec.attr.loc[2]
        attr3 = self.rec.attr.loc[3]

        self.assertEqual([1, 3], sorted(result))
        self.assertIn(1, attr1.ingredient_ids)
        self.assertNotIn(1, attr2.ingredient_ids)
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
    def test_top_n_const_3(self):
        result = self.rec.top_n_const_3(2, [1, 0, 1, 1])

        self.assertEqual([0, 1, 2], sorted(result))

        # modify weight for testing
        self.rec.c_alp = 1
        result = self.rec.top_n_const_3(2, [1, 0, 1, 1])

        self.assertEqual(0, result[2])

    # test top_n_const_3 slice results
    def test_top_n_const_3_2(self):
        # modify arguments for testing
        self.rec.result_N = 1
        self.rec.c_alp = 1
        result = self.rec.top_n_const_3(2, [1, 0, 1, 1])

        self.assertTrue((result[0] == 1) or (result[0] == 2))

    # test recommendation results for entire user
    def test_get_top_n(self):
        ret = self.rec.get_top_n()

        cols = [x for x in range(0, self.rec.result_N)]
        cols.extend(['i1', 'i2', 'nl'])

        self.assertEqual((4, 13), ret.shape)
        self.assertEqual(cols, list(ret.columns))
        self.assertEqual(4, len(ret))

    # evaluation using RMSE
    def test_RMSE(self):
        rec2 = post_rec.PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv', split=True)
        rec2.get_data()
        rec2.train()
        predictions = rec2.test_rmse()
        accuracy.rmse(predictions, False)

    # check post-processing in test_RMSE
    def test_RMSE2(self):
        rec2 = post_rec.PostRec('./data/rate.csv', './data/attr.csv', './data/const.csv', split=True)
        rec2.get_data()
        rec2.train()

        # Post-RS estimate each rating as 0
        rec2.test_RMSE_set = [('0', '2', 5.0), ('1', '2', 5.0), ('1', '2', 5.0)]
        predictions = rec2.test_rmse()
        self.assertEqual(5.0, accuracy.rmse(predictions, False))

        # Post-RS estimate ratings considering nutrient score = 1
        rec2.test_RMSE_set = [('2', '1', 5.0)]
        predictions = rec2.test_rmse()

        est_algo = rec2.algo.predict('2', '1')[3] * (1 - rec2.c_alp) + 5.0 * rec2.c_alp
        self.assertEqual(round(predictions[0][3], 5), round(est_algo, 5))

        # Post-RS estimate ratings same (No constraint)
        rec2.test_RMSE_set = [('3', '1', 5.0)]
        predictions = rec2.test_rmse()
        self.assertEqual(predictions[0][3], rec2.algo.predict('3', '1')[3])


if __name__ == '__main__':
    unittest.main()

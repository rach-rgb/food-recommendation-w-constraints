import unittest
from sys import path
from surprise import SVD
from surprise import accuracy

path.append('../src')
from src.rec_base import FoodRecBase
from src.load_data import load_rate_df


class TestRecBase(unittest.TestCase):
    # Dummy Recommendation System for testing
    class DummyRS(FoodRecBase):
        def top_n_const_1(self, uid, iid):
            return []

        def top_n_const_2(self, uid, iid):
            return []

        def top_n_const_3(self, uid, target):
            return []

        def get_top_n(self):
            return []

    def setUp(self):
        self.rec = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD())

    # initialization
    def test_init(self):
        self.assertEqual(self.rec.rate_file, './data/rate.csv')
        self.assertEqual(self.rec.attr_file, './data/attr.csv')
        self.assertEqual(self.rec.const_file, './data/const.csv')
        self.assertIsNotNone(self.rec.algo)
        self.assertFalse(self.rec.split)

    # get_data() collects required data properly
    def test_get_data(self):
        self.rec.get_data()

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

        # no test_RMSE_set when split if False
        self.assertIsNone(self.rec.test_RMSE_set)

    # get_data() when split is True
    def test_get_data2(self):
        rec2 = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), True)
        rec2.get_data()

        self.assertIsNotNone(rec2.test_RMSE_set)

    # train() and test()
    def test_inference(self):
        self.rec.get_data()
        self.rec.train()

        self.rec.test()
        self.assertIsNotNone(self.rec.predictions)

    # train() and test_rmse()
    def test_inference2(self):
        rec2 = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), True)
        rec2.get_data()
        rec2.train()
        predictions = rec2.test_rmse()
        accuracy.rmse(predictions, False)

    # get_constraint()
    def test_get_constraint(self):
        self.rec.get_data()
        const = self.rec.get_constraint(0)

        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], None)

        self.assertIsNone(self.rec.get_constraint(3))  # No constraint for user 3

    # test include_ingr
    def test_include_ingr(self):
        self.rec.get_data()

        self.assertTrue(self.rec.include_ingr(0, 1))
        self.assertFalse(self.rec.include_ingr(0, 4))
        self.assertFalse(self.rec.include_ingr(0, '1'))

    # test exclude_ingr
    def test_exclude_ingr(self):
        self.rec.get_data()

        self.assertTrue(self.rec.exclude_ingr(0, 4))
        self.assertFalse(self.rec.exclude_ingr(0, 1))
        self.assertTrue(self.rec.exclude_ingr(0, '1'))

    # test apply_nutr
    def test_apply_nutr(self):
        self.rec.get_data()

        list1 = [1, 0, 1, 1]
        list2 = [2, 0, 2, 2]
        list3 = [1, 0, 0, 0]

        self.assertEqual(0.67, round(self.rec.apply_nutr(0, list1), 2))
        self.assertEqual(0.67, round(self.rec.apply_nutr(0, list2), 2))
        self.assertEqual(1.00, round(self.rec.apply_nutr(1, list1), 2))
        self.assertEqual(1.00, round(self.rec.apply_nutr(1, list2), 2))
        self.assertEqual(0.00, round(self.rec.apply_nutr(0, list3), 2))

    # get_rel()
    def test_get_rel(self):
        self.rec.get_data()

        self.assertEqual(0, self.rec.get_rel(0, 2))
        self.assertEqual(1, self.rec.get_rel(0, 0))
        self.assertEqual(1, self.rec.get_rel(1, 0))
        self.assertEqual(0, self.rec.get_rel(1, 2))
        self.assertEqual(1.00, round(self.rec.get_rel(2, 1), 3))
        self.assertGreater(1, self.rec.get_rel(2, 0))
        self.assertEqual(1, self.rec.get_rel(3, 0))

    # read rate files from RS
    def test_save_rates(self):
        rec2 = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), True)
        rec2.get_data()
        # manipulate DummyRS.test_RMSE_set for testing
        rec2.test_RMSE_set = [('0', '0', 5.0), ('0', '1', 4.0), ('0', '2', 3.0), ('1', '0', 5.0), ('2', '1', 4.0),
                              ('2', '2', 2.0), ('3', '1', 4.0)]
        rec2.save_rates(rec2.test_RMSE_set, 'test')
        df = load_rate_df('../../result/test.csv')

        list0 = df.loc[0].iloc[0]
        list1 = df.loc[1].iloc[0]
        list2 = df.loc[2].iloc[0]
        list3 = df.loc[3].iloc[0]

        self.assertEqual([0, 1], sorted(list0))
        self.assertEqual([0], sorted(list1))
        self.assertEqual([1], sorted(list2))
        self.assertEqual([1], sorted(list3))

    # valid_constraint()
    def test_valid_constraint(self):
        self.rec.get_data()

        self.assertTrue(self.rec.valid_constraint(0, i1=1))
        self.assertTrue(self.rec.valid_constraint(1, i2=4))
        self.assertTrue(self.rec.valid_constraint(2, nl=[1, 0, 1, 1]))

        self.assertFalse(self.rec.valid_constraint(0, i1=1, i2=2))
        self.assertFalse(self.rec.valid_constraint(2, nl=[1, 0, 1, 0.5]))
        self.assertFalse(self.rec.valid_constraint(3))  # No constraint for user 3
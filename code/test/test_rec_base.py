import unittest
from sys import path
from surprise import SVD
from surprise import accuracy

path.append('../src')
from rec_base import FoodRecBase


class TestRecBase(unittest.TestCase):
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
            return []

    def setUp(self):
        self.rec = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD())
        self.rec_split = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), split=True)
        self.rec2 = self.DummyRS('./data/rate2.csv', './data/attr2.csv', './data/const2.csv', SVD())

    # initialization
    def test_init(self):
        self.assertEqual(self.rec.rate_file, './data/rate.csv')
        self.assertEqual(self.rec.attr_file, './data/attr.csv')
        self.assertEqual(self.rec.const_file, './data/const.csv')
        self.assertIsNotNone(self.rec.algo)
        self.assertFalse(self.rec.split)

    # get_data()
    def test_get_data(self):
        self.rec.get_data()

        # check attr data
        attr = self.rec.attr.loc[0]
        self.assertIn(1, attr.ingredient_ids)
        self.assertNotIn(4, attr.ingredient_ids)
        self.assertEqual(attr.nutrition[0], 0)

        # check const data
        const = self.rec.const.loc[self.rec.const.u == 0].iloc[0]
        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], None)

        # no test_RMSE_set when split is False
        self.assertIsNone(self.rec.test_RMSE_set)

    # get_data() when split is True
    def test_get_data2(self):
        self.rec_split.get_data()

        # create test_RMSE_set
        self.assertIsNotNone(self.rec_split.test_RMSE_set)

    # train() & test()
    def test_inference(self):
        self.rec.get_data()
        self.rec.train()
        self.rec.test()

        # create rec.predictions
        self.assertIsNotNone(self.rec.predictions)

    # train() & test_rmse()
    def test_inference2(self):
        self.rec_split.get_data()
        self.rec_split.train()
        predictions = self.rec_split.test_rmse()

        # return predictions
        accuracy.rmse(predictions, False)

    # get_constraint()
    def test_get_constraint(self):
        self.rec.get_data()
        const = self.rec.get_constraint(0)

        self.assertEqual(1, const['i1'])
        self.assertEqual(None, const['i2'])
        self.assertEqual(const['nl'], None)
        # No constraint for user 3
        self.assertIsNone(self.rec.get_constraint(3))

    # include_ingr()
    def test_include_ingr(self):
        self.rec.get_data()

        self.assertTrue(self.rec.include_ingr(0, 1))
        self.assertFalse(self.rec.include_ingr(0, 4))
        self.assertFalse(self.rec.include_ingr(0, '1'))

    # exclude_ingr()
    def test_exclude_ingr(self):
        self.rec.get_data()

        self.assertTrue(self.rec.exclude_ingr(0, 4))
        self.assertFalse(self.rec.exclude_ingr(0, 1))

    # apply_nutr()
    def test_apply_nutr(self):
        self.rec.get_data()

        def round_apply_nutr(fid, target):
            return round(self.rec.apply_nutr(fid, target), 2)

        self.assertEqual(0.67, round_apply_nutr(0, [1, 0, 1, 1]))
        self.assertEqual(0.67, round_apply_nutr(0, [2, 0, 2, 2]))
        self.assertEqual(1.00, round_apply_nutr(1, [1, 0, 1, 1]))
        self.assertEqual(1.00, round_apply_nutr(1, [2, 0, 2, 2]))
        self.assertEqual(0.00, round_apply_nutr(0, [1, 0, 0, 0]))
        # when numerator = 1
        self.assertEqual(0.00, round_apply_nutr(0, [0, 0, 0, 0]))

    # cal_rel() for single constraint
    def test_cal_rel(self):
        self.rec.get_data()

        self.assertEqual(0, self.rec.cal_rel(0, 2))
        self.assertEqual(1, self.rec.cal_rel(0, 0))
        self.assertEqual(1, self.rec.cal_rel(1, 0))
        self.assertEqual(0, self.rec.cal_rel(1, 2))
        self.assertEqual(1.00, round(self.rec.cal_rel(2, 1), 3))
        # 0 < score < 1
        self.assertLess(self.rec.cal_rel(2, 0), 1)
        self.assertLess(0, self.rec.cal_rel(2, 0))
        # no constraint
        self.assertEqual(1, self.rec.cal_rel(3, 0))

    # cal_rel() for mixed constraint
    def test_cal_rel2(self):
        self.rec2.get_data()

        self.assertEqual(1, self.rec2.cal_rel(3, 0))
        self.assertEqual(0, self.rec2.cal_rel(3, 1))
        self.assertEqual(0, self.rec2.cal_rel(3, 3))
        self.assertEqual(1, self.rec2.cal_rel(4, 0))
        self.assertEqual(0, self.rec2.cal_rel(4, 2))
        self.assertEqual(0, self.rec2.cal_rel(4, 1))
        self.assertEqual(1, self.rec2.cal_rel(5, 0))
        self.assertEqual(0, self.rec2.cal_rel(5, 2))
        self.assertEqual(0, self.rec2.cal_rel(5, 3))
        self.assertEqual(1, self.rec2.cal_rel(6, 0))
        self.assertEqual(0, self.rec2.cal_rel(6, 2))
        self.assertEqual(0, self.rec2.cal_rel(6, 3))

    # get_rel()
    def test_get_rel(self):
        self.rec_split = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), True)
        self.rec_split.get_data()

        rel_dict = self.rec_split.get_rel()
        for u in rel_dict.keys():
            for i in rel_dict[u]:
                self.assertGreaterEqual(self.rec.cal_rel(u, i), self.rec_split.rel_th)

    # read rate files from RS
    def test_save_rates(self):
        self.rec_split = self.DummyRS('./data/rate.csv', './data/attr.csv', './data/const.csv', SVD(), True)
        self.rec_split.get_data()
        # manipulate DummyRS.test_RMSE_set for testing
        self.rec_split.test_RMSE_set = [('0', '0', 5.0), ('0', '1', 4.0), ('0', '2', 3.0), ('1', '0', 5.0), ('2', '1', 4.0),
                              ('2', '2', 2.0), ('3', '1', 4.0)]
        rel_dict = self.rec_split.get_rel()

        # known_user: 0, 1, 3 & known_items: 1, 2
        # item '0' is unknown
        self.assertFalse(0 in rel_dict[0])
        self.assertEqual([1], rel_dict[0])
        # user '2' is unknown
        self.assertFalse(2 in rel_dict.keys())
        self.assertEqual([1], rel_dict[3])

    # valid_constraint() for single constraint
    def test_valid_constraint(self):
        self.rec.get_data()

        self.assertTrue(self.rec.valid_constraint(0, i1=1))
        self.assertTrue(self.rec.valid_constraint(1, i2=4))
        self.assertTrue(self.rec.valid_constraint(2, nl=[1, 0, 1, 1]))

        self.assertFalse(self.rec.valid_constraint(0, i1=1, i2=2))
        self.assertFalse(self.rec.valid_constraint(2, nl=[1, 0, 1, 0.5]))
        # No constraint for user 3
        self.assertTrue(self.rec.valid_constraint(3))
        self.assertFalse(self.rec.valid_constraint(3, i1=1))

    # valid_constraint() for mixed constraints
    def test_valid_constraint2(self):
        self.rec2.get_data()

        self.assertTrue(self.rec2.valid_constraint(3, i1=0, i2=1))
        self.assertFalse(self.rec2.valid_constraint(3, i1=0, i2=1, nl=[1, 1]))
        self.assertTrue(self.rec2.valid_constraint(4, i2=1, nl=[1, 1]))
        self.assertFalse(self.rec2.valid_constraint(4))
        self.assertTrue(self.rec2.valid_constraint(6, i1=0, i2=1, nl=[1, 1]))

from abc import *
import load_data as ld
from surprise.model_selection import train_test_split


# Interface of Food Recommendation System with Constraints
class FoodRecBase(metaclass=ABCMeta):
    result_N = 10  # get 10 result
    n_err = 5  # error bound for nutrient information

    def __init__(self, rate_file, attr_file, const_file, algo, split=False):
        # collect required file names
        self.rate_file = rate_file
        self.attr_file = attr_file
        self.const_file = const_file

        # algorithm to use
        self.algo = algo

        # required data
        self.attr = None
        self.const = None
        self.train_set = None
        self.test_set = None
        self.test_RMSE_set = None
        self.predictions = None

        # flags
        self.split = split

    def get_data(self):
        # get attribute data
        self.attr = ld.load_attr(self.attr_file)
        self.const = ld.load_const(self.const_file)

        # generate train_set and test_set
        data = ld.load_rate(self.rate_file)

        if self.split is True:  # split to test RMSE
            self.train_set, self.test_RMSE_set = train_test_split(data, test_size=0.25, random_state=42)
        else:
            self.train_set = data.build_full_trainset()
        self.test_set = self.train_set.build_anti_testset()

    # train with train_set
    def train(self):
        self.algo.fit(self.train_set)

    # make prediction with test_set
    def test(self):
        self.predictions = self.algo.test(self.test_set)

    # make prediction with test_RMSE
    def test_rmse(self):
        assert(self.split is True)

        return self.algo.test(self.test_set)

    # check if given constraints exist in const.file
    def valid_constraint(self, uid, i1=None, i2=None):
        const = self.const.loc[self.const.u == uid]
        if len(const) < 1:  # no constraint for user
            return False
        else:
            const = const.iloc[0]
        return (const['i1'] == i1) and (const['i2'] == i2)

        ##### To be implemented for const 3
        ## TODO

    # return list of top-N recommended food for uid s.t. includes iid
    @abstractmethod
    def top_n_const_1(self, uid, iid):
        pass

    # return list of top-N recommended food for uid s.t. excludes iid
    @abstractmethod
    def top_n_const_2(self, uid, iid):
        pass

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given iid as history
    @abstractmethod
    def top_n_const_3(self, uid, iid, target):
        pass

    # return recommendation and applied constants for entire user
    @abstractmethod
    def get_top_n(self):
        pass

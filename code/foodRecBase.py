from abc import *


# Interface of Food Recommendation System with Constraints
class FoodRecBase(metaclass=ABCMeta):
    top_N = 10
    # attributes
    # algo
    # train_set
    # test_set
    # predictions

    # train with train_set
    def train(self):
        self.algo.fit(self.train_set)

    # make prediction with test_set
    def test(self):
        self.predictions = self.algo.test(self.test_set)

    # load required data
    @abstractmethod
    def get_data(self):
        pass

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

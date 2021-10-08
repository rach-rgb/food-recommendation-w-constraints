from abc import *


class FoodRecBase(metaclass=ABCMeta):
    top_N = 10

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

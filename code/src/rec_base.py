from abc import *
from collections import defaultdict

from numpy import dot
from numpy.linalg import norm
from surprise.model_selection import train_test_split

import load_data as ld


# Interface of Food Recommendation System with Constraints
class FoodRecBase(metaclass=ABCMeta):
    result_N = 10  # get 10 result
    c_alp = 0.5  # weight for real number constraint
    rel_th = 0.5  # relevance threshold

    # constraint related columns
    c_i1 = 'i1'  # ingredient to be included
    c_i2 = 'i2'  # ingredient to be excluded
    c_nl = 'nl'  # nutrient target

    # Recommendation System Initialization
    def __init__(self, rate_file, attr_file, const_file, algo, need_test=False):
        # data file name
        self.rate_file = rate_file  # user-item rate pairs w/o constraint
        self.attr_file = attr_file  # item attributes
        self.const_file = const_file  # constraints for each user

        # prediction algorithm
        self.algo = algo

        # data
        self.attr = None
        self.const = None
        self.train_set = None
        self.predict_set = None  # anti-set of train_set
        self.test_set = None  # test set for RMSE evaluation
        self.predictions = None

        # generate test set for RMSE evaluation if need_test is True
        self.need_test = need_test

    # collects required data
    def get_data(self):
        # get attribute & constraint data
        self.attr = ld.load_attr(self.attr_file)
        self.const = ld.load_const(self.const_file)

        # generate train_set and predict_set
        data = ld.load_rate(self.rate_file)

        if self.need_test is True:  # split train set to create test_set
            self.train_set, self.test_set = train_test_split(data, test_size=0.25, random_state=42)
        else:
            self.train_set = data.build_full_trainset()
        self.predict_set = self.train_set.build_anti_testset()

    # train with train_set
    def train(self):
        self.algo.fit(self.train_set)

    # save prediction for predict_set at self.prediction
    def test(self):
        self.predictions = self.algo.test(self.predict_set)

    # return prediction for test_set
    def test_rmse(self):
        assert(self.need_test is True)

        return self.algo.test(self.test_set)

    # get constraint of user u, return None if there's no constraint
    def get_constraint(self, u):
        const = self.const.loc[self.const.u == u]
        if len(const) < 1:
            return None
        else:
            return const.iloc[0]

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.attr.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return iid not in self.attr.loc[fid].ingredient_ids

    # return score for constraint 3
    # 0 <= score <= 1
    def apply_nutr(self, fid, target):
        nutr = self.attr.loc[fid].nutrition
        d = dot(nutr, target)
        if d == 0:
            return 0
        return d / (norm(nutr) * norm(target))

    # get relevance score for (uid, iid)
    # 0 <= rel <= 1
    def cal_rel(self, uid, iid):
        const = self.get_constraint(uid)
        if const is None:
            return 1
        else:
            # apply constraint
            if const[self.c_i1] is not None:
                if not self.include_ingr(iid, const[self.c_i1]):
                    return 0
            if const[self.c_i2] is not None:
                if not self.exclude_ingr(iid, const[self.c_i2]):
                    return 0
            if const[self.c_nl] is not None:
                adder = self.apply_nutr(iid, const[self.c_nl])
                return round(adder, 1)
            return 1

    # return (user, item, rate) dictionary
    # s.t (relevance of user-item) >= rel_th, where default value of rel_th is 4
    def get_rel(self):
        rel = defaultdict(list)

        for (u, i, r) in self.test_set:
            if r >= 4 and self.cal_rel(int(u), int(i)) >= self.rel_th:
                if u in self.train_set._raw2inner_id_users.keys():
                    if i in self.train_set._raw2inner_id_items.keys():
                        rel[int(u)].append(int(i))

        return rel

    # check if given constraints exist in const.file
    def valid_constraint(self, uid, i1=None, i2=None, nl=None):
        const = self.get_constraint(uid)
        if const is None:
            return (i1 is None) and (i2 is None) and (nl is None)

        return (const[self.c_i1] == i1) and (const[self.c_i2] == i2) and (const[self.c_nl] == nl)

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
    def top_n_const_3(self, uid, target):
        pass

    # return list of top-N recommended food for uid
    # s.t satisfies specific constraint
    @abstractmethod
    def top_n_const(self, uid, iid1, iid2, target):
        pass

    # return recommendation and applied constants for entire user
    @abstractmethod
    def get_top_n(self):
        pass

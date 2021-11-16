from abc import *
from collections import defaultdict
from pandas import DataFrame
from numpy import dot
from numpy.linalg import norm
from surprise.model_selection import train_test_split
import load_data as ld


# Interface of Food Recommendation System with Constraints
class FoodRecBase(metaclass=ABCMeta):
    result_N = 10  # get 10 result
    c_alp = 0.5  # weight for constraint
    rel_th = 0.5 # relevance threshold

    # Constraint Related Columns
    c_i1 = 'i1'  # ingredient to be included
    c_i2 = 'i2'  # ingredient to be excluded
    c_nl = 'nl'  # nutrient target

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

        return self.algo.test(self.test_RMSE_set)

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
        return dot(nutr, target) / (norm(nutr) * norm(target))

    # get relevance score for (uid, iid)
    def cal_rel(self, uid, iid):
        const = self.get_constraint(uid)
        if const is None:
            return 1
        else:
            # apply constraint
            # assume there's only one constraint TODO
            if const[self.c_i1] is not None:
                if not self.include_ingr(iid, const[self.c_i1]):
                    return 0
            elif const[self.c_i2] is not None:
                if not self.exclude_ingr(iid, const[self.c_i2]):
                    return 0
            elif const[self.c_nl] is not None:
                adder = self.apply_nutr(iid, const[self.c_nl])
                return adder
            return 1

    # save (user, item, rate) list as dataframe
    def get_rel(self):
        rates = defaultdict(list)
        for (u, i, r) in self.test_RMSE_set:
            if r >= 4 and self.cal_rel(int(u), int(i)) >= self.rel_th:
                if u in self.train_set._raw2inner_id_users.keys():
                    if i in self.train_set._raw2inner_id_items.keys():
                        rates[int(u)].append(int(i))

        return rates

    # check if given constraints exist in const.file
    def valid_constraint(self, uid, i1=None, i2=None, nl=None):
        const = self.const.loc[self.const.u == uid]
        if len(const) < 1:  # no constraint for user
            return False
        else:
            const = const.iloc[0]
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

    # return recommendation and applied constants for entire user
    @abstractmethod
    def get_top_n(self):
        pass

from tood_rec_base import *
import load_data as ld
from surprise import SVD
from collections import defaultdict


class PostRec(FoodRecBase):

    def __init__(self, rate_file, attr_file, const_file, nutr_error=5, algo=SVD()):
        # set constants
        self.candidate = 30
        self.nutr_error = nutr_error

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

        # result
        self.predictions = None
        self.top_K = None

    def get_data(self):
        # get attribute data
        self.attr = ld.load_attr(self.attr_file)
        self.const = ld.load_const(self.const_file)

        # generate train_set and test_set
        data = ld.load_rate(self.rate_file)
        self.train_set = data.build_full_trainset()
        self.test_set = self.train_set.build_anti_testset()

    # modify source form https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    def sort_prediction(self):
        # First map the predictions to each user.
        self.top_K = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            self.top_K[int(uid)].append((int(iid), est))  # get uid and iid as integer

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in self.top_K.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)

    # return list of top-N recommended food for uid s.t. includes iid
    def top_n_const_1(self, uid, iid):
        if not self.valid_constraint(uid, i1=iid):
            return []

        result = []
        for f in self.top_K[uid]:
            if self.include_ingr(f[0], iid):
                result.append(f[0])
            if len(result) >= self.top_N:
                break

        return result

    # return list of top-N recommended food for uid s.t. excludes iid
    def top_n_const_2(self, uid, iid):
        if not self.valid_constraint(uid, i2=iid):
            return []

        result = []
        for f in self.top_K[uid]:
            if self.exclude_ingr(f[0], iid):
                result.append(f[0])
            if len(result) >= self.top_N:
                break

        return result

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given fid
    def top_n_const_3(self, uid, iid, target):
        result = []
        for f in self.top_K[str(uid)]:
            if self.satisfy_nutr(int(f[0]), iid, target):
                result.append(int(f[0]))
            if len(result) >= self.top_N:
                break

        return result

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.attr.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return not iid in self.attr.loc[fid].ingredient_ids

    # return T/F for constraint 3
    def satisfy_nutr(self, fid, hid, target_nutr):
        nutr = self.attr.loc[fid].nutrition
        nutr_hist = self.attr.loc[hid].nutrition
        for i in range(0, len(nutr)):
            target = target_nutr[i] - float(nutr_hist[i])
            value = float(nutr[i])
            if value < (target * (1 - (self.nutr_error / 100))):
                return False
            if (target * (1 + (self.nutr_error / 100))) < value:
                return False
        return True

    # check if given constraints exist in const.file
    def valid_constraint(self, uid, i1=None, i2=None):
        const = self.const.loc[self.const.u == uid]
        if len(const) < 1:  # no constraint for user
            return False
        else:
            const = const.iloc[0]
        return (const['i1'] == i1) and (const['i2'] == i2)

        ##### To be implemented for const 3

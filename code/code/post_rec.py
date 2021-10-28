from collections import defaultdict
import pandas as pd
from surprise import SVD
from rec_base import *


class PostRec(FoodRecBase):

    def __init__(self, rate_file, attr_file, const_file, algo=SVD()):
        super().__init__(rate_file, attr_file, const_file, algo)

        # result
        self.top_K = None  # dictionary of [(item, rate)] for each user

    # modify source form https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    # sort prediction and get top_K
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
        if self.top_K is None:
            self.sort_prediction()

        result = []
        for f in self.top_K[uid]:
            if self.include_ingr(f[0], iid):
                result.append(f[0])
            if len(result) >= self.result_N:
                break

        return result

    # return list of top-N recommended food for uid s.t. excludes iid
    def top_n_const_2(self, uid, iid):
        if not self.valid_constraint(uid, i2=iid):
            return []
        if self.top_K is None:
            self.sort_prediction()

        result = []
        for f in self.top_K[uid]:
            if self.exclude_ingr(f[0], iid):
                result.append(f[0])
            if len(result) >= self.result_N:
                break

        return result

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given fid
    # TODO
    def top_n_const_3(self, uid, iid, target):
        result = []
        for f in self.top_K[str(uid)]:
            if self.satisfy_nutr(int(f[0]), iid, target):
                result.append(int(f[0]))
            if len(result) >= self.result_N:
                break

        return result

    # return recommendation and applied constants for entire user
    def get_top_n(self):
        if self.top_K is None:
            self.sort_prediction()

        top_N = defaultdict(list)
        for u in self.top_K.keys():
            const = self.const.loc[self.const.u == u]
            if len(const) < 1:  # no constraint
                top_N[u] = [i for i, r in self.top_K[u]]
                continue
            const = const.iloc[0]

            # assume there's only constraint
            # TODO
            if const['i1'] is not None:
                top_N[u] = self.top_n_const_1(u, const['i1'])
            elif const['i2'] is not None:
                top_N[u] = self.top_n_const_2(u, const['i2'])

        result = pd.DataFrame.from_dict(top_N, orient='index')
        result = result.reindex(columns=[x for x in range(0, self.result_N)])
        return result.join(self.const.set_index('u'))

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.attr.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return iid not in self.attr.loc[fid].ingredient_ids

    # return T/F for constraint 3
    # TODO
    def satisfy_nutr(self, fid, hid, target_nutr):
        nutr = self.attr.loc[fid].nutrition
        nutr_hist = self.attr.loc[hid].nutrition
        for i in range(0, len(nutr)):
            target = target_nutr[i] - float(nutr_hist[i])
            value = float(nutr[i])
            if value < (target * (1 - (self.n_err / 100))):
                return False
            if (target * (1 + (self.n_err / 100))) < value:
                return False
        return True

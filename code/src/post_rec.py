from collections import defaultdict
import heapq

import pandas as pd
from numpy import dot
from numpy.linalg import norm
from surprise import SVD

from rec_base import *


class PostRec(FoodRecBase):

    def __init__(self, rate_file, attr_file, const_file, algo=SVD(), split=False):
        super().__init__(rate_file, attr_file, const_file, algo, split)

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
    # s.t satisfies target nutrient
    def top_n_const_3(self, uid, target):
        if not self.valid_constraint(uid, nl=target):
            return []
        if self.top_K is None:
            self.sort_prediction()

        c3_rates = []  # (rate, ID)
        for f in self.top_K[uid][:self.result_N]:
            new_rate = self.apply_nutr(f[0], target) * self.c_alp * 5
            new_rate = new_rate + (1 - self.c_alp) * f[1]
            heapq.heappush(c3_rates, (new_rate, f[0]))

        for f in self.top_K[uid][self.result_N:]:
            if (1 - self.c_alp) * f[1] + 5 * self.c_alp < c3_rates[0][0]:
                break

            new_rate = self.apply_nutr(f[0], target) * self.c_alp * 5
            new_rate = new_rate + (1 - self.c_alp) * f[1]
            if new_rate > c3_rates[0][0]:
                heapq.heappop(c3_rates)
                heapq.heappush(c3_rates, (new_rate, f[0]))

        result = []
        while len(c3_rates) > 0:
            result.append(heapq.heappop(c3_rates)[1])

        result.reverse()

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
            if const[self.c_i1] is not None:
                top_N[u] = self.top_n_const_1(u, const[self.c_i1])
            elif const[self.c_i2] is not None:
                top_N[u] = self.top_n_const_2(u, const[self.c_i2])
            elif const[self.c_nl] is not None:
                top_N[u] = self.top_n_const_3(u, const[self.c_nl])

        result = pd.DataFrame.from_dict(top_N, orient='index')
        result = result.reindex(columns=[x for x in range(0, self.result_N)])
        return result.join(self.const.set_index('u'))

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

    # make prediction with test_RMSE set
    def test_rmse(self):
        assert(self.split is True)

        # test_RMSE_set: (str(u), str(i), real rate) tuple

        return self.algo.test(self.test_RMSE_set)
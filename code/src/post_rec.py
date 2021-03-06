import heapq
import pandas as pd
from surprise import SVD
from surprise.prediction_algorithms.predictions import Prediction

from rec_base import *


# This recommendation system applies constraints after training
class PostRec(FoodRecBase):
    def __init__(self, rate_file, attr_file, const_file, algo=SVD(), need_test=False):
        super().__init__(rate_file, attr_file, const_file, algo, need_test)

        # result
        self.top_K = None  # dictionary of [(item, rate)] for each user

    # return prediction with test_RMSE set
    def test_rmse(self):
        assert(self.need_test is True)

        pre = self.algo.test(self.test_set)
        for i in range(0, len(pre)):
            uid, iid, true_r, est, _ = pre[i]

            const = self.get_constraint(int(uid))
            if const is None:
                continue

            # apply constraint
            new_est = est
            if const[self.c_i1] is not None:
                if not self.include_ingr(int(iid), const[self.c_i1]):
                    new_est = 0.0  # make est = 0
            if new_est != 0.0 and const[self.c_i2] is not None:
                if not self.exclude_ingr(int(iid), const[self.c_i2]):
                    new_est = 0.0
            if new_est != 0.0 and const[self.c_nl] is not None:
                adder = self.apply_nutr(int(iid), const[self.c_nl])
                new_est = est * (1-self.c_alp) + self.c_alp * 5 * adder

            if new_est != est:
                pre[i] = Prediction(uid, iid, true_r, new_est, _)

        return pre

    # sort prediction and get top_K
    def sort_prediction(self):
        self.top_K = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            self.top_K[int(uid)].append((int(iid), est))

        for uid, user_ratings in self.top_K.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)

    # return list of top-N recommended food for uid s.t. includes iid
    # uid, iid is integer
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

    # return list of top-N recommended food for uid
    # s.t satisfies specific constraint
    def top_n_const(self, uid, iid1=None, iid2=None, target=None):
        if not self.valid_constraint(uid, iid1, iid2, target):
            return []
        if self.top_K is None:
            self.sort_prediction()

        candidates = []  # (rates, ID)
        check = 0
        for f in self.top_K[uid]:
            check = check + 1
            rate = f[1]
            if iid1 is not None:
                if not self.include_ingr(f[0], iid1):
                    continue
            if iid2 is not None:
                if not self.exclude_ingr(f[0], iid2):
                    continue
            if target is not None:
                adder = self.apply_nutr(f[0], target) * self.c_alp * 5
                rate = adder + (1 - self.c_alp) * f[1]
            heapq.heappush(candidates, (rate, f[0]))

            if len(candidates) == self.result_N:
                break

        if target is not None:
            for f in self.top_K[uid][check:]:
                if (1 - self.c_alp) * f[1] + 5 * self.c_alp < candidates[0][0]:
                    break

                if iid1 is not None:
                    if not self.include_ingr(f[0], iid1):
                        continue
                if iid2 is not None:
                    if not self.exclude_ingr(f[0], iid2):
                        continue

                new_rate = self.apply_nutr(f[0], target) * self.c_alp * 5
                new_rate = new_rate + (1 - self.c_alp) * f[1]
                if new_rate > candidates[0][0]:
                    heapq.heappop(candidates)
                    heapq.heappush(candidates, (new_rate, f[0]))

        result = []
        while len(candidates) > 0:
            result.append(heapq.heappop(candidates)[1])

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

            top_N[u] = self.top_n_const(u, const[self.c_i1], const[self.c_i2], const[self.c_nl])

        result = pd.DataFrame.from_dict(top_N, orient='index')
        result = result.reindex(columns=[x for x in range(0, self.result_N)])
        return result.join(self.const.set_index('u')).sort_index()


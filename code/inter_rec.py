from collections import defaultdict
import pandas as pd
from rec_base import *


def get_items(items):
    ret = []
    for i, r in items:
        ret.append(int(i))
    return ret


class InterRec(FoodRecBase):

    def __init__(self, rate_file, attr_file, const_file, algo):
        super().__init__(rate_file, attr_file, const_file, algo)

        # result
        self.top_N = None  # dictionary of [item] for each user

    def get_data(self):
        super().get_data()

        self.algo.set_data(self.attr, self.const, self.n_err)

    # sort prediction and get top_N directly
    def sort_predictions(self):
        top_n_rate = defaultdict(list)
        self.top_N = defaultdict(list)

        # First map the predictions to each user.
        for uid, iid, true_r, est, _ in self.predictions:
            top_n_rate[int(uid)].append((int(iid), est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n_rate.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n_rate[uid] = user_ratings[:self.result_N]
            self.top_N[uid] = [i for i, r in top_n_rate[uid]]

    def get_top_n(self):
        if self.top_N is None:
            self.sort_predictions()

        result = pd.DataFrame.from_dict(self.top_N, orient='index')
        result = result.reindex(columns=[x for x in range(0, self.result_N)])
        return result.join(self.const.set_index('u'))

    # return list of top-N recommended food for uid s.t. includes iid
    def top_n_const_1(self, uid, iid):
        if not self.valid_constraint(uid, i1=iid):
            return []
        if self.top_N is None:
            self.sort_predictions()

        return self.top_N[uid]

    # return list of top-N recommended food for uid s.t. excludes iid
    def top_n_const_2(self, uid, iid):
        if not self.valid_constraint(uid, i2=iid):
            return []
        if self.top_N is None:
            self.sort_predictions()

        return self.top_N[uid]

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given fid
    # TODO
    def top_n_const_3(self, uid, iid, target):
        id = self.map_uid(uid, None, None, iid, target)
        return get_items(self.top_n[id])
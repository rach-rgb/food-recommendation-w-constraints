import load_data as ld
from foodRecBase import *
from collections import defaultdict


def get_items(items):
    ret = []
    for i, r in items:
        ret.append(int(i))
    return ret


class AttrRec(FoodRecBase):
    # attributes from FoodRecBase
    # algo
    # train_set
    # test_set
    # predictions

    # additional attributes
    # recipe_data
    # candidate
    # nutr_error
    # top_k
    # self.const

    def __init__(self, algo, candidate=30, nutr_error=5, const_path='../data/const_data.csv'):
        # set constants
        self.candidate = candidate
        self.nutr_error = nutr_error
        self.algo = algo
        self.const_path = const_path

    def get_data(self):
        data = ld.load_reduced_rating_data()
        self.const = ld.load_const(self.const_path)

        self.train_set = data.build_full_trainset()
        self.test_set = self.train_set.build_anti_testset()

        # get recipe data
        self.recipe_data = ld.load_recipe_data()

        self.algo.set_data(self.recipe_data, self.const)

    # source form https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    def get_candidate(self):
        n = self.top_N
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        self.top_n = top_n

    # find uid with constraint
    def map_uid(self, uid, i1, i2, h, n):

        return uid

    # return list of top-N recommended food for uid s.t. includes iid
    def top_n_const_1(self, uid, iid):
        id = self.map_uid(uid, iid, None, None, None)
        return get_items(self.top_n[id])

    # return list of top-N recommended food for uid s.t. excludes iid
    def top_n_const_2(self, uid, iid):
        id = self.map_uid(uid, None, iid, None, None)
        return get_items(self.top_n[id])

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given fid
    def top_n_const_3(self, uid, iid, target):
        id = self.map_uid(uid, None, None, iid, target)
        return get_items(self.top_n[id])

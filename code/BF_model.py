import load_data as ld
from foodRecBase import *
from surprise.model_selection import train_test_split
from surprise import SVD
from collections import defaultdict


class BFRec(FoodRecBase):
    # recipe_data
    # train_set, test_set
    # algo
    # predictions
    # top_k

    def __init__(self, candidate=30, nutr_error=5):
        # set constants
        self.candidate = candidate
        self.nutr_error = nutr_error

    # load required data
    def get_data(self):
        # get recipe data
        self.recipe_data = ld.load_recipe_data()

        # generate train_set and test_set
        data = ld.load_reduced_rating_data()
        self.train_set = data.build_full_trainset()
        self.test_set = self.train_set.build_anti_testset()
        # self.train_set, self.test_set = train_test_split(data, test_size=.25)

    def train(self):
        self.algo = SVD()
        self.algo.fit(self.train_set)

    def test(self):
        self.predictions = self.algo.test(self.test_set)

    # source form https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    def get_candidate(self):
        n = self.candidate
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        self.top_k = top_n

    # return list of top-N recommended food for uid s.t. includes iid
    def top_n_const_1(self, uid, iid):
        result = []
        for f in self.top_k[str(uid)]:
            if self.include_ingr(int(f[0]), iid):
                result.append(int(f[0]))
            if len(result) >= self.top_N:
                break

        return result

    # return list of top-N recommended food for uid s.t. excludes iid
    def top_n_const_2(self, uid, iid):
        result = []
        for f in self.top_k[str(uid)]:
            if self.exclude_ingr(int(f[0]), iid):
                result.append(int(f[0]))
            if len(result) >= self.top_N:
                break

        return result

    # return list of top-N recommended food for uid
    # s.t satisfied target nutrient given fid
    def top_n_const_3(self, uid, iid, target):
        result = []
        for f in self.top_k[str(uid)]:
            if self.satisfy_nutr(int(f[0]), iid, target):
                result.append(int(f[0]))
            if len(result) >= self.top_N:
                break

        return result

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.recipe_data.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return not iid in self.recipe_data.loc[fid].ingredient_ids

    # return T/F for constraint 3
    def satisfy_nutr(self, fid, hid, target_nutr):
        nutr = self.recipe_data.loc[fid].nutrition
        nutr_hist = self.recipe_data.loc[hid].nutrition
        for i in range(0, len(nutr)):
            target = target_nutr[i] - float(nutr_hist[i])
            value = float(nutr[i])
            if value < (target * (1 - (self.nutr_error / 100))):
                return False
            if (target * (1 + (self.nutr_error / 100))) < value:
                return False
        return True

import numpy as np
from surprise import SVD
from surprise import PredictionImpossible


class SVD_TF(SVD):
    def __init__(self):

        # Always call base method before doing anything.
        SVD.__init__(self)

    # pandas dataframe related to attribute and constraints
    def set_data(self, attr_data, const_data):
        self.item_attr = attr_data
        self.const = const_data
        self.nutr_error = 5

    # u, i inner id
    # raw_u, raw_i raw id
    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        raw_u = int(self.trainset.to_raw_uid(u))
        raw_i = int(self.trainset.to_raw_iid(i))
        constraint = self.const.loc[raw_u]

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        if not self.check_constraint(raw_i, constraint):
            return 0
        else:
            return est

    # return T/F for constraint 1
    def check_constraint(self, raw_i, constraint):
        x = (constraint.i1 is None) or self.include_ingr(raw_i, constraint.i1)
        if x is False: return x
        y = (constraint.i2 is None) or self.exclude_ingr(raw_i, constraint.i2)
        if y is False: return y
        z = (constraint.hl is None) or self.satisfy_nutr(raw_i, constraint.hl, constraint.nl)
        return z

    # return T/F for constraint 1
    def include_ingr(self, fid, iid):
        return iid in self.item_attr.loc[fid].ingredient_ids

    # return T/F for constraint 2
    def exclude_ingr(self, fid, iid):
        return not iid in self.item_attr.loc[fid].ingredient_ids

    # return T/F for constraint 3
    def satisfy_nutr(self, fid, hid, target_nutr):
        nutr = self.item_attr.loc[fid].nutrition
        nutr_hist = self.item_attr.loc[hid].nutrition
        for i in range(0, len(nutr)):
            target = target_nutr[i] - float(nutr_hist[i])
            value = float(nutr[i])
            if value < (target * (1 - (self.nutr_error / 100))):
                return False
            if (target * (1 + (self.nutr_error / 100))) < value:
                return False
        return True

